from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import sqlite3
from typing import Optional, Any

from keras.models import load_model, Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit uploads to 16MB

# --- Load model once globally ---
model: Optional[Model] = None
try:
    loaded: Any = load_model("face_emotionModel.h5")
    if isinstance(loaded, Model):
        model = loaded
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Loaded object is not a Keras Model")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Emotion labels ---
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Emotion comments ---
emotion_comments = {
    'Angry': "Calm down boss 😤 — who provoke you like this?",
    'Disgust': "Eww 😖... you sure say you dey okay like this?",
    'Fear': "Omo! You resemble person we just see rat now now 😳",
    'Happy': "Ahh see joy! 😄 — Whatever you’re doing, keep it up!",
    'Sad': "Egbon! Why you dey squeeze face like this?? 😢 — Smile small abeg!",
    'Surprise': "Wetin happen 😲 — shey DSA don ban starch shirt???",
    'Neutral': "Expressionless 🤨 — normal straight face nau!"
}

# --- Initialize SQLite database ---
def init_db() -> None:
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            emotion TEXT,
            comment TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Home route ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# --- Prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    name: str = request.form.get('name', '').strip()
    file = request.files.get('file')

    if not name or file is None:
        return jsonify({"error": "Name or file missing"}), 400

    # Save file
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = file.filename or "uploaded_image.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and preprocess image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    # Check model
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Predict
    predictions = model.predict(img)
    pred_idx = int(np.argmax(predictions))
    emotion = emotion_labels[pred_idx]
    comment = emotion_comments.get(emotion, "No comment available")

    # Save to database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO emotions (name, emotion, comment) VALUES (?, ?, ?)",
        (name, emotion, comment)
    )
    conn.commit()
    conn.close()

    # Return result page
    return render_template('result.html', name=name, emotion=emotion, comment=comment, image=filename)

# --- Optional health check endpoint for Render ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Use production server in Render
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
