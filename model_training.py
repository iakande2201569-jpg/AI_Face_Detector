'''
# model_training.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical  # ✅ ensure TF namespace

# -----------------------------
# Load FER2013 CSV
# -----------------------------
def load_fer2013(csv_path, image_size=(48, 48), test_size=0.1, val_size=0.1, random_state=42):
    """
    Load FER2013 CSV and return (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    print(f"Loading FER2013 from {csv_path} ...")

    df = pd.read_csv(csv_path)
    if 'pixels' not in df.columns:
        raise ValueError("CSV does not contain 'pixels' column. Check the file format.")

    pixels = df['pixels'].tolist()
    n = len(pixels)
    h, w = image_size
    X = np.zeros((n, h, w), dtype=np.uint8)

    for i, px_str in enumerate(pixels):
        arr = np.fromstring(px_str, dtype=int, sep=' ')
        if arr.size != h * w:
            raise ValueError(f"Row {i} has {arr.size} pixels, expected {h*w}")
        X[i] = arr.reshape((h, w))

    if 'emotion' not in df.columns:
        raise ValueError("CSV does not contain 'emotion' column.")
    y = df['emotion']

    # Normalize and expand dims
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, -1)

    # One-hot encode
    num_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes=num_classes)

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=random_state, stratify=y
    )
    val_fraction_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=random_state,
        stratify=np.argmax(y_temp, axis=1)
    )

    print("Data split sizes:", X_train.shape, X_val.shape, X_test.shape)
    print("Loaded FER2013 successfully.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# -----------------------------
# Model Definition
# -----------------------------
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
)
from keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model(input_shape=(48, 48, 1), num_classes=7):
    """Build a Convolutional Neural Network (CNN) for facial emotion recognition."""
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# Training Script

if __name__ == "__main__":
    csv_path = os.path.join("data", "fer2013.csv")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(csv_path)

    model = build_model(input_shape=(48, 48, 1), num_classes=y_train.shape[1])

    
    data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

    checkpoint = ModelCheckpoint("face_emotionModel.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

   
    history = model.fit(
        data_augmentation(X_train),  # augment images on the fly
        y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=64,
        callbacks=[checkpoint, early_stop]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {test_acc:.2f}")

    model.save("face_emotionModel.h5")
    print("Model saved as face_emotionModel.h5 ✅")


'''



