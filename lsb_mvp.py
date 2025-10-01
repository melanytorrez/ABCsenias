#!/usr/bin/env python3
"""
MVP: Reconocer letras (estáticas) de la LSB con landmarks de MediaPipe.
Modo consola con tres subcomandos:
  - collect : captura muestras etiquetadas y las guarda en data/lsb_alpha.csv
  - train   : entrena un clasificador con las muestras y guarda models/lsb_alpha.joblib
  - run     : ejecuta el reconocimiento en vivo desde la cámara
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
DEFAULT_LABELS = [
    "a","b","c","d","e","f","g","h","i","k","l","ll","m","n","o","p","q","r","s","t","u","v","w","x","y"
]

DATA_CSV = Path("data/lsb_alpha.csv")
MODEL_PATH = Path("models/lsb_alpha.joblib")
DATA_SEQ_CSV = Path("data/lsb_seq.csv")
MODEL_SEQ_PATH = Path("models/lsb_seq.joblib")

SEQ_LEN = 20  # << --- CORRECCIÓN: Definido aquí, al principio.

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
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts_rel = pts - wrist
    base_idxs = [5, 9, 13, 17]
    dists = [np.linalg.norm(pts[i] - wrist) for i in base_idxs]
    hand_size = (np.mean(dists) + 1e-6)
    pts_rel /= hand_size
    return pts_rel.reshape(-1)


@dataclass
class HandFrame:
    has_hand: bool
    features: np.ndarray | None
    frame: np.ndarray | None


def read_hand_frame(cap, draw=False) -> HandFrame:
    ok, frame = cap.read()
    if not ok:
        return HandFrame(False, None, None)
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
            if draw:
                mp_drawing.draw_landmarks(
                    frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )
            return HandFrame(True, feats, frame if draw else None)
        else:
            return HandFrame(False, None, frame if draw else None)


# ---------------------------- Subcomando: collect (MODIFICADO) ----------------------------

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

    # --- NUEVA LÓGICA DE CAPTURA ---
    current_label_to_capture = None
    last_capture_time = 0
    CAPTURE_INTERVAL_SECONDS = 0.2  # Intervalo para no guardar demasiadas fotos iguales

    print("\n[CONTROLES]")
    print("  - Presiona una letra para INICIAR la captura continua de esa seña.")
    print("  - Presiona OTRA letra para cambiar a una nueva seña.")
    print("  - Presiona ESPACIO para DETENER la captura actual.")
    print("  - Presiona ';' para capturar 'll'.")
    print("  - Presiona ESC para salir y guardar.")

    special_map = { ord(";"): "ll" }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame no capturado.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats = None

            with mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as hands:
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0].landmark
                    feats = landmarks_to_features(lms)
                    mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, "MANO DETECTADA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO HAY MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- LÓGICA DE TECLADO Y CAPTURA AUTOMÁTICA ---
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord(' '):  # Barra espaciadora para detener
                if current_label_to_capture:
                    print(f"[INFO] Captura de '{current_label_to_capture}' detenida.")
                    current_label_to_capture = None
            elif k in special_map:
                label = special_map[k]
                if current_label_to_capture != label:
                    print(f"[INFO] Iniciando captura para '{label}'")
                    current_label_to_capture = label
            elif 32 <= k <= 126:
                ch = chr(k).lower()
                if ch in labels:
                    if current_label_to_capture != ch:
                        print(f"[INFO] Cambiando captura a '{ch}'")
                        current_label_to_capture = ch
                else:
                     print(f"[WARN] Tecla '{ch}' no está en etiquetas permitidas.")

            # Captura automática si hay una etiqueta seleccionada y se detecta una mano
            if current_label_to_capture and feats is not None:
                current_time = time.time()
                if (current_time - last_capture_time) > CAPTURE_INTERVAL_SECONDS:
                    row = dict({f"f{i}": v for i, v in enumerate(feats)}, label=current_label_to_capture)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    print(f"[OK] Guardada muestra para '{current_label_to_capture}'. Total={len(df)}")
                    last_capture_time = current_time
                    if len(df) % 20 == 0:  # Guarda el progreso cada 20 muestras
                        df.to_csv(DATA_CSV, index=False)

            # Muestra el estado actual en la pantalla
            if current_label_to_capture:
                status_text = f"CAPTURANDO: '{current_label_to_capture.upper()}'"
                color = (0, 255, 255)
            else:
                status_text = "MODO: ESPERA"
                color = (255, 255, 0)
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Recolección de datos estáticos", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if not df.empty:
            df.to_csv(DATA_CSV, index=False)
            print(f"[INFO] Dataset guardado en {DATA_CSV} ({len(df)} muestras).")

# ---------------------------- Subcomando: collect-seq (dinámicas) ----------------------------

class SequenceBuffer:
    def __init__(self, maxlen=SEQ_LEN):
        self.frames = deque(maxlen=maxlen)
    def add(self, feats):
        if feats is not None:
            self.frames.append(feats)
    def is_ready(self):
        return len(self.frames) == self.frames.maxlen
    def get_sequence(self):
        return np.concatenate(self.frames)

def cmd_collect_seq(args):
    labels = ["j", "ñ", "rr", "z"]
    print(f"[INFO] Colección de secuencias para: {labels}")

    DATA_SEQ_CSV.parent.mkdir(parents=True, exist_ok=True)
    if DATA_SEQ_CSV.exists():
        df = pd.read_csv(DATA_SEQ_CSV)
        print(f"[INFO] Cargando dataset dinámico existente con {len(df)} muestras.")
    else:
        df = pd.DataFrame()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        sys.exit(1)

    buffer = SequenceBuffer(maxlen=SEQ_LEN)
    print("[CONTROLES] Presiona la tecla de la letra (j, ñ, rr, z) para GUARDAR la secuencia. ESC para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as hands:
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0].landmark
                    feats = landmarks_to_features(lms)
                    buffer.add(feats)
                    mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                    if buffer.is_ready():
                        cv2.putText(frame, "LISTO PARA CAPTURAR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("collect-seq", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if 32 <= k <= 126:
                ch = chr(k).lower()
                if ch in labels:
                    if buffer.is_ready():
                        row = dict({f"f{i}": v for i, v in enumerate(buffer.get_sequence())}, label=ch)
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        print(f"[OK] Secuencia para '{ch}' guardada. Total={len(df)}")
                        if len(df) % 5 == 0:
                            df.to_csv(DATA_SEQ_CSV, index=False)
                    else:
                        print("[WARN] Buffer no está lleno. Realiza el gesto más despacio.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if not df.empty:
            df.to_csv(DATA_SEQ_CSV, index=False)
            print(f"[INFO] Dataset dinámico guardado en {DATA_SEQ_CSV} ({len(df)} muestras).")

# ---------------------------- Subcomando: train ----------------------------
def cmd_train(args):
    if not DATA_CSV.exists():
        print(f"[ERROR] No existe {DATA_CSV}. Ejecuta 'collect' primero.")
        sys.exit(1)

    df = pd.read_csv(DATA_CSV).dropna()
    print(f"[INFO] Dataset cargado con {len(df)} muestras tras limpiar NaN.")
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

# ---------------------------- Subcomando: train-seq (dinámicas) ----------------------------
def cmd_train_seq(args):
    if not DATA_SEQ_CSV.exists():
        print("[ERROR] No existe dataset dinámico.")
        sys.exit(1)

    df = pd.read_csv(DATA_SEQ_CSV).dropna()
    print(f"[INFO] Dataset dinámico cargado con {len(df)} muestras tras limpiar NaN.")
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])

    print("[INFO] Entrenando modelo secuencial...")
    clf.fit(X_train, y_train)
    print(f"[RESULT] Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.3f}")

    MODEL_SEQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "labels": sorted(np.unique(y)), "seq_len": SEQ_LEN}, MODEL_SEQ_PATH)
    print(f"[INFO] Modelo secuencial guardado en {MODEL_SEQ_PATH}")

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
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
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
                    pred_smooth, _ = Counter(hist).most_common(1)[0]
                    msg = f"{pred_smooth} ({conf:.2f})"
                    if args.show:
                        mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                else:
                    if args.show:
                        cv2.putText(frame, "NO HAY MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if args.show:
                cv2.imshow("run - reconocimiento", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------- Subcomando: run-seq (dinámicas) ----------------------------
def cmd_run_seq(args):
    if not MODEL_SEQ_PATH.exists():
        print("[ERROR] No existe modelo dinámico.")
        sys.exit(1)
    bundle = joblib.load(MODEL_SEQ_PATH)
    clf: Pipeline = bundle["pipeline"]
    seq_len = bundle.get("seq_len", SEQ_LEN)

    cap = cv2.VideoCapture(args.cam)
    buffer = SequenceBuffer(maxlen=seq_len)
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with mp_hands.Hands(max_num_hands=1) as hands:
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    feats = landmarks_to_features(res.multi_hand_landmarks[0].landmark)
                    buffer.add(feats)
                    if buffer.is_ready():
                        X = buffer.get_sequence().reshape(1, -1)
                        proba = clf.predict_proba(X)[0]
                        pred = clf.classes_[np.argmax(proba)]
                        cv2.putText(frame, f"{pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("run-seq", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------- Subcomando: run-hybrid (auto estático/dinámico mejorado) ----------------------------
def motion_score(prev_feats, curr_feats):
    if prev_feats is None or curr_feats is None:
        return 0.0
    return np.linalg.norm(curr_feats - prev_feats)

def cmd_run_hybrid(args):
    if not MODEL_PATH.exists() or not MODEL_SEQ_PATH.exists():
        print("[ERROR] Faltan modelos. Entrena con 'train' y 'train-seq'.")
        sys.exit(1)

    bundle_static = joblib.load(MODEL_PATH)
    clf_static: Pipeline = bundle_static["pipeline"]
    bundle_seq = joblib.load(MODEL_SEQ_PATH)
    clf_seq: Pipeline = bundle_seq["pipeline"]
    seq_len = bundle_seq.get("seq_len", SEQ_LEN)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        sys.exit(1)

    buffer = SequenceBuffer(maxlen=seq_len)
    prev_feats = None
    motion_hist = deque(maxlen=15)
    hist_static = deque(maxlen=10) # Para suavizar predicción estática

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
                res = hands.process(rgb)
                msg = ""
                if res.multi_hand_landmarks:
                    feats = landmarks_to_features(res.multi_hand_landmarks[0].landmark)
                    if prev_feats is not None and feats is not None:
                        score = motion_score(prev_feats, feats)
                        motion_hist.append(score)
                    prev_feats = feats

                    if feats is not None:
                        moving = sum(s > args.motion_threshold for s in motion_hist)
                        if moving >= args.motion_frames:
                            buffer.add(feats)
                            msg = "[DINAMICO...]"
                            if buffer.is_ready():
                                X = buffer.get_sequence().reshape(1, -1)
                                proba = clf_seq.predict_proba(X)[0]
                                conf = np.max(proba)
                                if conf >= args.min_conf:
                                    pred = clf_seq.classes_[np.argmax(proba)]
                                    msg = f"[D] {pred} ({conf:.2f})"
                        else:
                            hist_static.clear() # Limpiar buffer estático si hubo movimiento
                            X = feats.reshape(1, -1)
                            proba = clf_static.predict_proba(X)[0]
                            pred = clf_static.classes_[np.argmax(proba)]
                            conf = np.max(proba)
                            msg = f"[E] {pred} ({conf:.2f})"

                    if args.show:
                        mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                
                if args.show:
                    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.imshow("run-hybrid", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------- Main CLI ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Reconocimiento de Lengua de Señas")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Capturar señas estáticas de forma continua")
    p_collect.add_argument("--labels", type=str, default=None, help="Lista de etiquetas separadas por comas")
    p_collect.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    p_collect.set_defaults(func=cmd_collect)

    p_collect_seq = sub.add_parser("collect-seq", help="Capturar señas dinámicas (j, ñ, rr, z)")
    p_collect_seq.add_argument("--cam", type=int, default=0)
    p_collect_seq.set_defaults(func=cmd_collect_seq)

    p_train = sub.add_parser("train", help="Entrenar el modelo para señas estáticas")
    p_train.set_defaults(func=cmd_train)

    p_train_seq = sub.add_parser("train-seq", help="Entrenar el modelo para señas dinámicas")
    p_train_seq.set_defaults(func=cmd_train_seq)

    p_run = sub.add_parser("run", help="Reconocimiento en tiempo real (solo estático)")
    p_run.add_argument("--cam", type=int, default=0)
    p_run.add_argument("--smooth", type=int, default=10, help="Ventana para estabilizar predicción")
    p_run.add_argument("--show", action="store_true", help="Mostrar ventana con esqueleto")
    p_run.set_defaults(func=cmd_run)

    p_run_seq = sub.add_parser("run-seq", help="Reconocimiento en vivo (solo dinámico)")
    p_run_seq.add_argument("--cam", type=int, default=0)
    p_run_seq.set_defaults(func=cmd_run_seq)

    p_run_hybrid = sub.add_parser("run-hybrid", help="Reconocimiento híbrido (estático/dinámico)")
    p_run_hybrid.add_argument("--cam", type=int, default=0)
    p_run_hybrid.add_argument("--show", action="store_true", help="Mostrar ventana de visualización")
    p_run_hybrid.add_argument("--motion-threshold", type=float, default=0.15, help="Umbral de movimiento para activar modo dinámico")
    p_run_hybrid.add_argument("--motion-frames", type=int, default=8, help="Nº de frames con movimiento para activar modo dinámico")
    p_run_hybrid.add_argument("--min-conf", type=float, default=0.7, help="Confianza mínima para predicción dinámica")
    p_run_hybrid.set_defaults(func=cmd_run_hybrid)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()