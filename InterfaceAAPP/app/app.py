from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import shutil
import atexit

# Initialisation de l'application Flask
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Nécessaire pour utiliser les sessions

# Charger le modèle YOLO
model_path = r"C:\Users\edoua\Desktop\InterfaceAAPP\app\model\best_011224.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# Charger le fichier Excel
file_path = r"C:\Users\edoua\Desktop\Comprehensive_Risk_Context_Table.xlsx"
try:
    df_combined = pd.read_excel(file_path, sheet_name='Risques_Complexes')
    df_simple = pd.read_excel(file_path, sheet_name='Risques_Simples')
except Exception as e:
    print(f"Error loading Excel file: {e}")

# Routes Flask
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Récupérer les informations utilisateur
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        user_id = request.form.get("user_id")

        # Enregistrer les données utilisateur dans la session
        session["user"] = {
            "first_name": first_name,
            "last_name": last_name,
            "user_id": user_id
        }

        # Créer un répertoire pour l'utilisateur
        user_folder = f"static/outputs/{first_name}_{last_name}_{user_id}"
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        return redirect(url_for("upload_page"))
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    # Vérifier si l'utilisateur est connecté
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("upload.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Vérifiez si l'utilisateur est connecté
    if "user" not in session:
        return jsonify({"error": "User information missing"}), 400

    # Récupérer les informations utilisateur
    user = session["user"]
    user_folder = f"static/outputs/{user['first_name']}_{user['last_name']}_{user['user_id']}"

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")

        # Validation de la résolution de l'image
        if image.size != (1920, 1080):
            return jsonify({"error": "Image resolution must be exactly 1920x1080."}), 400

    except Exception as e:
        return jsonify({"error": f"Error loading image: {e}"}), 400

    # YOLO prediction
    try:
        with torch.no_grad():
            results = model(image)
    except Exception as e:
        return jsonify({"error": f"Error during YOLO prediction: {e}"}), 500

    # Extract predictions
    predictions = []
    detected_classes = []
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        draw = ImageDraw.Draw(image)
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            width = x_max - x_min
            height = y_max - y_min
            class_name = results[0].names[int(box.cls[0])]
            detected_classes.append(class_name)
            confidence = f"{float(box.conf[0]) * 100:.2f}%"
            predictions.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [x_min, y_min, width, height]
            })
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            draw.text((x_min, y_min - 10), f"{class_name} {confidence}", fill="red")

    # Associate risks
    risks = get_risks(detected_classes)

    # Save the annotated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image_path = os.path.join(user_folder, f"annotated_image_{timestamp}.jpg")
    image.save(output_image_path)

    # Save predictions and risks to a file
    output_data_path = os.path.join(user_folder, f"predictions_{timestamp}.json")
    output_data = {
        "predictions": predictions,
        "risks": risks,
        "image_path": output_image_path
    }
    with open(output_data_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    return jsonify({"predictions": predictions, "risks": risks, "image_path": output_image_path})


def get_risks(detected_classes):
    detected_set = set(detected_classes)
    combined_risks = []

    for _, row in df_combined.iterrows():
        combination = {row['Class 1'], row['Class 2']}
        if combination.issubset(detected_set):
            combined_risks.extend(row['Combined Risks'].split(", "))

    if combined_risks:
        return combined_risks

    individual_risks = []
    for cls in detected_classes:
        match = df_simple[df_simple['Class'] == cls]
        if not match.empty:
            individual_risks.append(match.iloc[0]['Risk'])

    return individual_risks if individual_risks else ["No specific risks identified"]


# Nettoyage automatique des fichiers après 24 heures
def cleanup_old_files():
    base_folder = "static/outputs"
    now = datetime.now()

    if not os.path.exists(base_folder):
        return

    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            # Vérifiez l'âge du dossier
            creation_time = datetime.fromtimestamp(os.path.getctime(folder_path))
            if now - creation_time > timedelta(hours=24):
                try:
                    shutil.rmtree(folder_path)  # Supprimez tout le dossier
                    print(f"Deleted old folder: {folder_path}")
                except Exception as e:
                    print(f"Error deleting folder {folder_path}: {e}")


# Planification de la tâche avec APScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=cleanup_old_files,
    trigger=IntervalTrigger(hours=1),
    id='cleanup_job',
    name='Cleanup old files every hour',
    replace_existing=True
)
scheduler.start()


atexit.register(lambda: scheduler.shutdown())


if __name__ == "__main__":
    if not os.path.exists("static/outputs"):
        os.makedirs("static/outputs")
    app.run(debug=True)
