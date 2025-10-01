# lsb_mvp.py (Versión Limpia para Collect y Train)

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Importamos la lógica compartida desde nuestro nuevo archivo de utilidades
import lsb_mvp_utils as utils

# --- Configuración Base ---
DEFAULT_LABELS = ["a","b","c","d","e","f","g","h","i","k","l","ll","m","n","o","p","q","r","s","t","u","v","w","x","y"]
DATA_CSV = Path("data/lsb_alpha.csv")
MODEL_PATH = Path("models/lsb_alpha.joblib")
DATA_SEQ_CSV = Path("data/lsb_seq.csv")
MODEL_SEQ_PATH = Path("models/lsb_seq.joblib")
SEQ_LEN = 20
RANDOM_STATE = 42

# --- Subcomando: collect (Estático) ---
def cmd_collect(args):
    # ... (El código de esta función no necesita cambiar, pero ahora usará 'utils')
    labels = args.labels.split(",") if args.labels else DEFAULT_LABELS
    labels = [x.strip().lower() for x in labels if x.strip()]
    print(f"[INFO] Colección de datos para etiquetas: {labels}")

    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV) if DATA_CSV.exists() else pd.DataFrame()
    if not df.empty:
        print(f"[INFO] Cargando dataset existente con {len(df)} muestras.")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        sys.exit(1)

    current_label_to_capture = None
    last_capture_time = 0
    CAPTURE_INTERVAL_SECONDS = 0.2

    print("\n[CONTROLES]\n  - Presiona una letra para INICIAR la captura continua.\n  - Presiona ESPACIO para DETENER la captura.\n  - Presiona ';' para 'll'.\n  - Presiona ESC para salir y guardar.")
    special_map = { ord(";"): "ll" }

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)

            # Usamos el MediaPipe de las utilidades para dibujar y obtener landmarks
            with utils.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                feats = None
                if res.multi_hand_landmarks:
                    utils.mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], utils.mp_hands.HAND_CONNECTIONS)
                    feats = utils.landmarks_to_features(res.multi_hand_landmarks[0].landmark)
                    cv2.putText(frame, "MANO DETECTADA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO HAY MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            elif k == ord(' '):
                if current_label_to_capture:
                    print(f"[INFO] Captura de '{current_label_to_capture}' detenida.")
                    current_label_to_capture = None
            elif k in special_map:
                current_label_to_capture = special_map[k]
                print(f"[INFO] Iniciando captura para '{current_label_to_capture}'")
            elif 32 <= k <= 126:
                ch = chr(k).lower()
                if ch in labels:
                    current_label_to_capture = ch
                    print(f"[INFO] Cambiando captura a '{ch}'")

            if current_label_to_capture and feats is not None and (time.time() - last_capture_time) > CAPTURE_INTERVAL_SECONDS:
                row = dict({f"f{i}": v for i, v in enumerate(feats)}, label=current_label_to_capture)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                print(f"[OK] Guardada muestra para '{current_label_to_capture}'. Total={len(df)}")
                last_capture_time = time.time()
                if len(df) % 20 == 0: df.to_csv(DATA_CSV, index=False)
            
            status_text = f"CAPTURANDO: '{current_label_to_capture.upper()}'" if current_label_to_capture else "MODO: ESPERA"
            color = (0, 255, 255) if current_label_to_capture else (255, 255, 0)
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Recolección de datos estáticos", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if not df.empty:
            df.to_csv(DATA_CSV, index=False)
            print(f"[INFO] Dataset guardado en {DATA_CSV} ({len(df)} muestras).")


# --- Subcomando: collect-seq (Dinámico) ---
def cmd_collect_seq(args):
    # Usamos la clase SequenceBuffer importada desde utils
    buffer = utils.SequenceBuffer(maxlen=SEQ_LEN)
    # ... el resto del código es muy similar ...
    labels = ["j", "ñ", "rr", "z"]
    print(f"[INFO] Colección de secuencias para: {labels}")

    DATA_SEQ_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_SEQ_CSV) if DATA_SEQ_CSV.exists() else pd.DataFrame()
    if not df.empty:
        print(f"[INFO] Cargando dataset dinámico existente con {len(df)} muestras.")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened(): sys.exit("[ERROR] No se pudo abrir la cámara.")
    
    print("[CONTROLES] Presiona la tecla (j, ñ, rr, z) para GUARDAR la secuencia. ESC para salir.")

    try:
        while True:
            ok, frame = cap.read();
            if not ok: break
            frame = cv2.flip(frame, 1)

            with utils.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    feats = utils.landmarks_to_features(res.multi_hand_landmarks[0].landmark)
                    buffer.add(feats)
                    utils.mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], utils.mp_hands.HAND_CONNECTIONS)
                    if buffer.is_ready():
                        cv2.putText(frame, "LISTO PARA CAPTURAR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow("collect-seq", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if 32 <= k <= 126:
                ch = chr(k).lower()
                if ch in labels and buffer.is_ready():
                    row = dict({f"f{i}": v for i, v in enumerate(buffer.get_sequence())}, label=ch)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    print(f"[OK] Secuencia para '{ch}' guardada. Total={len(df)}")
                    if len(df) % 5 == 0: df.to_csv(DATA_SEQ_CSV, index=False)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if not df.empty:
            df.to_csv(DATA_SEQ_CSV, index=False)
            print(f"[INFO] Dataset dinámico guardado en {DATA_SEQ_CSV} ({len(df)} muestras).")


# --- Subcomandos: train y train-seq (Sin cambios) ---
def cmd_train(args):
    if not DATA_CSV.exists(): sys.exit(f"[ERROR] No existe {DATA_CSV}. Ejecuta 'collect' primero.")
    df = pd.read_csv(DATA_CSV).dropna()
    print(f"[INFO] Dataset estático cargado con {len(df)} muestras.")
    X = df.drop("label", axis=1).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, C=5.0, gamma="scale", random_state=RANDOM_STATE))])
    print("[INFO] Entrenando modelo estático...")
    clf.fit(X_train, y_train)
    print(f"[RESULT] Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.3f}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "labels": sorted(np.unique(y))}, MODEL_PATH)
    print(f"[INFO] Modelo guardado en {MODEL_PATH}")

def cmd_train_seq(args):
    if not DATA_SEQ_CSV.exists(): sys.exit("[ERROR] No existe dataset dinámico.")
    df = pd.read_csv(DATA_SEQ_CSV).dropna()
    print(f"[INFO] Dataset dinámico cargado con {len(df)} muestras.")
    X = df.drop("label", axis=1).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])
    print("[INFO] Entrenando modelo secuencial...")
    clf.fit(X_train, y_train)
    print(f"[RESULT] Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.3f}")
    MODEL_SEQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "labels": sorted(np.unique(y)), "seq_len": SEQ_LEN}, MODEL_SEQ_PATH)
    print(f"[INFO] Modelo secuencial guardado en {MODEL_SEQ_PATH}")

# --- Main CLI (Ahora solo con collect y train) ---
def main():
    parser = argparse.ArgumentParser(description="Herramienta de datos para Reconocimiento de Señas")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Capturar señas estáticas de forma continua")
    p_collect.add_argument("--labels",type=str,default=None)
    p_collect.add_argument("--cam", type=int, default=0)
    p_collect.set_defaults(func=cmd_collect)

    p_collect_seq = sub.add_parser("collect-seq", help="Capturar señas dinámicas")
    p_collect_seq.add_argument("--cam", type=int, default=0)
    p_collect_seq.set_defaults(func=cmd_collect_seq)

    p_train = sub.add_parser("train", help="Entrenar el modelo estático")
    p_train.set_defaults(func=cmd_train)

    p_train_seq = sub.add_parser("train-seq", help="Entrenar el modelo dinámico")
    p_train_seq.set_defaults(func=cmd_train_seq)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()