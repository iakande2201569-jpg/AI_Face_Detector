from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import sqlite3
from typing import Optional, Any

# ✅ Import TensorFlow and Keras Model explicitly
from keras.models import load_model, Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# --- Load the trained model once ---
model: Optional[Model] = None
try:
    loaded: Any = load_model("face_emotionmodel.h5")
    if isinstance(loaded, Model):
        model = loaded
    else:
        print("⚠️ Loaded object is not a Keras Model instance.")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Emotion labels ---
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Fun / cheeky comments for each emotion
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
    cursor.execute('''CREATE TABLE IF NOT EXISTS emotions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        emotion TEXT,
                        comment TEXT
                    )''')
    
    try:
        cursor.execute("ALTER TABLE emotions ADD COLUMN comment TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists, ignore error
    conn.commit()
    conn.close()

init_db()


# --- Home Route ---
@app.route('/')
def home():
    return render_template('index.html')


# --- Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    name: str = request.form.get('name', '').strip()
    file = request.files.get('file')

    if not name or file is None:
        return redirect(url_for('home'))

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename: str = file.filename or "uploaded_image.jpg"
    filepath: str = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and preprocess image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Error: Could not read image", 400

    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    # Ensure model is loaded
    if model is None:
        return "Model not loaded properly", 500

    # Predict emotion
    predictions = model.predict(img)
    pred_idx = int(np.argmax(predictions))
    emotion = emotion_labels[pred_idx]
    comment = emotion_comments.get(emotion, "No comment available")

    print(f"Predicted: {emotion} | Comment: {comment}")

# Save to SQLite
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO emotions (name, emotion, comment) VALUES (?, ?, ?)",
        (name, emotion, comment)
    )
    conn.commit()
    conn.close()

    return render_template('result.html', name=name, emotion=emotion, comment=comment, image=filename)


if __name__ == '__main__':
    app.run(debug=True)
