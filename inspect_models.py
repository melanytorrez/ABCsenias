import joblib
import numpy as np

def inspect_model(path):
    print(f"--- Inspecting {path} ---")
    data = joblib.load(path)
    pipeline = data["pipeline"]
    svc = pipeline.named_steps["svm"]
    scaler = pipeline.named_steps["scaler"]
    
    print(f"Kernel: {svc.kernel}")
    print(f"Number of classes: {len(svc.classes_)}")
    print(f"Classes: {svc.classes_}")
    print(f"Number of support vectors: {len(svc.support_vectors_)}")
    print(f"Feature shape: {svc.support_vectors_.shape[1]}")
    print(f"Scaler mean shape: {scaler.mean_.shape}")

inspect_model("models/lsb_alpha.joblib")
inspect_model("models/lsb_seq.joblib")
