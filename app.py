import os
import threading
import time
from datetime import datetime
import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import requests
import schedule
import fitz  # PyMuPDF
import google.generativeai as genai
from google.generativeai.types import content_types
from PIL import Image
import io
# === SETUP ===

app = Flask(__name__)
CORS(app)

# Gemini setup
genai.configure(api_key="AIzaSyBORbTjaV4axhDCtV5Z2yKDUsBMWUsTiCw")  # Replace with env variable in prod

# Pain level model and scaler
pain_model = load_model("model/pain_level_classifier_model.h5")
scaler = joblib.load("model/scaler.pkl")
label_map = {0: "mild", 1: "moderate", 2: "severe"}

# Upload path
UPLOAD_FOLDER = "./temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Backend URL for meds
BACKEND_URL = 'http://localhost:5000/medications'

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Memory in MB

@app.route("/memory-usage", methods=["GET"])
def memory_usage():
    mem_usage = get_memory_usage()
    return jsonify({"memory_usage_MB": round(mem_usage, 2)})

# === 1. PAIN LEVEL PREDICTION ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[data["baseline_fhr"], data["variability"], data["accelerations"], data["decelerations"]]])
    scaled = scaler.transform(features)
    prediction = pain_model.predict(scaled)
    label = label_map[np.argmax(prediction)]
    return jsonify({"pain_level": label})

# === 2. MEDICATION REMINDER SCHEDULER ===
def check_meds():
    try:
        meds = requests.get(BACKEND_URL).json()
        now = datetime.now().strftime('%H:%M')
        for med in meds:
            if not med['taken'] and med['time'] == now:
                print(f"🔔 Reminder: Take {med['name']} ({med['dosage']}) at {med['time']}")
    except Exception as e:
        print("Error checking meds:", e)

schedule.every(1).minutes.do(check_meds)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# === 3. APPOINTMENT EMAIL GENERATOR ===
@app.route('/generate-email', methods=['POST'])
def generate_email():
    data = request.get_json()
    message = f"""
    Hello {data.get('userName')},

    This is a gentle reminder for your upcoming appointment with Dr. {data.get('doctorName')} scheduled at {data.get('time')}.

    Please be on time and bring any necessary documents.

    Stay healthy!
    - HealthBot
    """
    return jsonify({"message": message.strip()})

# === 4. ANOMALY SCAN ANALYZER ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    ext = file.filename.split(".")[-1].lower()

    if ext == "pdf":
        text = extract_text_from_pdf(filepath)
        prompt = f"""
        You are a medical assistant AI. Given the following ultrasound or anomaly scan report:
        ------
        {text}
        ------
        Please do the following:
        1. Identify if the baby appears normal or if there is any anomaly.
        2. If there's an anomaly, explain it briefly.
        3. Provide simple health advice for the mother.
        4. Respond in a human-friendly tone for non-medical users.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return jsonify({"result": response.text.strip()})

    elif ext in ["jpg", "jpeg", "png", "webp"]:
        model = genai.GenerativeModel("gemini-1.5-flash")
        image = Image.open(filepath)
        prompt = "Please analyze the medical scan image and provide a user-friendly report with anomaly detection and health advice for the mother."
        response = model.generate_content([image, prompt])
        return jsonify({"result": response.text.strip()})

    else:
        return jsonify({"error": "Unsupported file type"}), 400

# === 5. STATUS CHECK ===
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"message": "✅ AI Agent (Gemini) + Scheduler + Pain Classifier Running"})

# === RUN SERVER + SCHEDULER ===
if __name__ == '__main__':
    threading.Thread(target=run_scheduler).start()
    app.run(host="0.0.0.0", port=8080)
