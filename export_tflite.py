import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os

def train_and_export_tflite(csv_path, tflite_path, labels_path, is_sequence=False):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"--- Training {tflite_path} ---")
    df = pd.read_csv(csv_path).dropna()
    
    # Feature scaling is often required for Neural Networks, but TFLite models
    # in mobile often perform better if they consume unscaled data (or we scale it manually).
    # Since landmarks_to_features already does some normalization (relative to hand size/wrist),
    # we might be able to skip explicit StandardScaler, or we must implement it in Android.
    # To keep it simple for Android, we train without StandardScaler and rely on the MediaPipe normalization.
    
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Encode labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Save labels mapping
    labels_dict = {str(i): label for i, label in enumerate(le.classes_)}
    with open(labels_path, "w") as f:
        json.dump(labels_dict, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    input_dim = X.shape[1]

    # Create a simple Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {tflite_path}\n")

if __name__ == "__main__":
    os.makedirs("models_tflite", exist_ok=True)
    
    # Train static model
    train_and_export_tflite(
        csv_path="data/lsb_alpha.csv",
        tflite_path="models_tflite/lsb_alpha.tflite",
        labels_path="models_tflite/lsb_alpha_labels.json",
        is_sequence=False
    )

    # Train dynamic model
    train_and_export_tflite(
        csv_path="data/lsb_seq.csv",
        tflite_path="models_tflite/lsb_seq.tflite",
        labels_path="models_tflite/lsb_seq_labels.json",
        is_sequence=True
    )
