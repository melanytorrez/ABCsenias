#!/usr/bin/env python3
"""
MVP: Reconocer letras (estáticas) de la LSB con landmarks de MediaPipe.
Modo consola con tres subcomandos:
  - collect : captura muestras etiquetadas y las guarda en data/lsb_alpha.csv
  - train   : entrena un clasificador con las muestras y guarda models/lsb_alpha.joblib
  - run     : ejecuta el reconocimiento en vivo desde la cámara
Autoría: Tu equipo + ChatGPT (MVP educativo).
"""
from __future__ import annotations
import argparse
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


# ---------------------------- Configuración base ----------------------------

# Fase 1: sólo letras con formas "estáticas" (sin trayectoria/animación).
# (Más adelante añadiremos j, ñ, rr, z con modelos temporales).
DEFAULT_LABELS = [
    "a","b","c","d","e","f","g","h","i","k","l","ll","m","n","o","p","q","r","s","t","u","v","w","x","y"
]

DATA_CSV = Path("data/lsb_alpha.csv")
MODEL_PATH = Path("models/lsb_alpha.joblib")
RANDOM_STATE = 42


# ---------------------------- Utilidades de visión ----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def landmarks_to_features(landmarks: list[mp.framework.formats.landmark_pb2.NormalizedLandmark]) -> np.ndarray:
    """
    Convierte los 21 landmarks a un vector de características robusto a escala y traslación.
    - Origen en la muñeca (landmark 0)
    - Se normaliza por el tamaño de la mano (distancia muñeca - MCP medio)
    - Devuelve (21 * 3) = 63 características: x_rel, y_rel, z_rel por punto.
    """
    if landmarks is None or len(landmarks) != 21:
        return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (21,3), normalizados [0..1]
    wrist = pts[0].copy()
    pts_rel = pts - wrist  # traslación
    # tamaño mano: promedio de distancias muñeca->(MCP índice=5, medio=9, anular=13, meñique=17)
    base_idxs = [5, 9, 13, 17]
    dists = [np.linalg.norm(pts[i] - wrist) for i in base_idxs]
    hand_size = (np.mean(dists) + 1e-6)
    pts_rel /= hand_size  # escala
    return pts_rel.reshape(-1)  # (63,)


@dataclass
class HandFrame:
    has_hand: bool
    features: np.ndarray | None
    frame: np.ndarray | None  # para visualización opcional


def read_hand_frame(cap, draw=False) -> HandFrame:
    ok, frame = cap.read()
    if not ok:
        return HandFrame(False, None, None)
    frame = cv2.flip(frame, 1)  # espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            feats = landmarks_to_features(lms)
            if draw:
                mp_drawing.draw_landmarks(
                    frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )
            return HandFrame(True, feats, frame if draw else None)
        else:
            return HandFrame(False, None, frame if draw else None)


# ---------------------------- Subcomando: collect ----------------------------

def cmd_collect(args):
    labels = args.labels.split(",") if args.labels else DEFAULT_LABELS
    labels = [x.strip().lower() for x in labels if x.strip()]
    print(f"[INFO] Colección de datos para etiquetas: {labels}")

    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        print(f"[INFO] Cargando dataset existente con {len(df)} muestras.")
    else:
        df = pd.DataFrame()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        sys.exit(1)

    print("[CONTROLES] Presiona la tecla de la letra para guardar (ej: 'a'). ESC para salir.")
    print("            Para etiquetas de dos caracteres (ll), presiona ';' para alternar a 'll'.")
    current_combo = None

    # Mapea algunas teclas a etiquetas de dos letras (simple hack para consola latina)
    special_map = {
        ord(";"): "ll",
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame no capturado.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0].landmark
                    feats = landmarks_to_features(lms)
                    mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, "MANO DETECTADA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    feats = None
                    cv2.putText(frame, "NO HAY MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("collect - muestra y presiona la letra", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            if k in special_map:
                current_combo = special_map[k]
                print(f"[INFO] Etiqueta compuesta seleccionada: {current_combo}")
                continue
            if 32 <= k <= 126:  # teclas imprimibles básicas
                ch = chr(k).lower()

                # preferir combo si está activo
                label = current_combo if current_combo else ch
                current_combo = None

                if label in labels:
                    if feats is not None:
                        row = dict({f"f{i}": v for i, v in enumerate(feats)}, label=label)
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        if len(df) % 10 == 0:
                            df.to_csv(DATA_CSV, index=False)
                        print(f"[OK] Guardada muestra para '{label}'. Total={len(df)}")
                    else:
                        print("[WARN] No se detectó mano; no se guardó muestra.")
                else:
                    print(f"[WARN] Tecla '{label}' no está en etiquetas permitidas: {labels}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if len(df):
            df.to_csv(DATA_CSV, index=False)
            print(f"[INFO] Dataset guardado en {DATA_CSV} ({len(df)} muestras).")


# ---------------------------- Subcomando: train ----------------------------

def cmd_train(args):
    if not DATA_CSV.exists():
        print(f"[ERROR] No existe {DATA_CSV}. Ejecuta 'collect' primero.")
        sys.exit(1)
    df = pd.read_csv(DATA_CSV)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, C=5.0, gamma="scale", random_state=RANDOM_STATE))
    ])

    print("[INFO] Entrenando SVM...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Accuracy de validación: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "labels": sorted(np.unique(y))}, MODEL_PATH)
    print(f"[INFO] Modelo guardado en {MODEL_PATH}")


# ---------------------------- Subcomando: run ----------------------------

def cmd_run(args):
    if not MODEL_PATH.exists():
        print(f"[ERROR] No existe el modelo {MODEL_PATH}. Entrena con 'train'.")
        sys.exit(1)
    bundle = joblib.load(MODEL_PATH)
    clf: Pipeline = bundle["pipeline"]

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        sys.exit(1)

    hist = deque(maxlen=args.smooth)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Sin frame.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0].landmark
                    feats = landmarks_to_features(lms).reshape(1, -1)
                    proba = clf.predict_proba(feats)[0]
                    idx = int(np.argmax(proba))
                    pred = clf.classes_[idx]
                    conf = float(proba[idx])

                    hist.append(pred)
                    # mayoría simple para estabilizar
                    pred_smooth, count = Counter(hist).most_common(1)[0]
                    msg = f"{pred_smooth} ({conf:.2f})"
                    if args.show:
                        mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                        cv2.imshow("run - reconocimiento", frame)
                    else:
                        # Modo consola puro: imprime cuando cambia la predicción estable
                        if len(hist) == hist.maxlen and hist[-1] != hist[-2]:
                            print(msg)

                else:
                    if args.show:
                        cv2.putText(frame, "NO HAY MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        cv2.imshow("run - reconocimiento", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------- Main CLI ----------------------------

def main():
    parser = argparse.ArgumentParser(description="MVP LSB -> Texto (alfabeto estático)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Capturar muestras etiquetadas con la cámara")
    p_collect.add_argument("--labels", type=str, default=None, help="Lista separada por comas (por defecto conjunto estático)")
    p_collect.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    p_collect.set_defaults(func=cmd_collect)

    p_train = sub.add_parser("train", help="Entrenar el modelo con las muestras")
    p_train.set_defaults(func=cmd_train)

    p_run = sub.add_parser("run", help="Reconocimiento en tiempo real")
    p_run.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    p_run.add_argument("--smooth", type=int, default=10, help="Ventana para estabilizar la predicción")
    p_run.add_argument("--show", action="store_true", help="Mostrar ventana con esqueleto de mano")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
