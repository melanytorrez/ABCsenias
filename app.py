# app.py (Versión Kiosco Local para Windows Totem)

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import time
from collections import deque
import numpy as np
import lsb_mvp_utils as utils
from threading import Lock
import joblib
from pathlib import Path
import sys
import os
import re

import engineio.async_drivers.threading
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

# --- Resolución de Rutas para PyInstaller ---
def resource_path(relative_path):
    """Obtiene la ruta absoluta de un recurso, compatible con desarrollo y empaquetado EXE."""
    try:
        # PyInstaller crea una carpeta temporal y guarda la ruta en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Inicializar Flask con ruta de plantillas dinámica
app = Flask(__name__, template_folder=resource_path("templates"))
app.config['SECRET_KEY'] = 'totem_secret_key_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Rutas de Modelos ---
MODEL_PATH = Path(resource_path("models/lsb_alpha.joblib"))
MODEL_SEQ_PATH = Path(resource_path("models/lsb_seq.joblib"))

clf_static = None
clf_seq = None

def load_models():
    """Carga los modelos entrenados al iniciar la aplicación."""
    global clf_static, clf_seq
    try:
        bundle_static = joblib.load(MODEL_PATH)
        clf_static = bundle_static["pipeline"]
        
        bundle_seq = joblib.load(MODEL_SEQ_PATH)
        clf_seq = bundle_seq["pipeline"]
        print("[INFO] Modelos de IA cargados correctamente para Windows Totem.")
    except Exception as e:
        print(f"\n[ERROR] No se pudieron cargar los modelos en {MODEL_PATH}: {e}")
        print("Asegúrate de haber copiado las carpetas models/ y models_tflite/ correctamente.")
        clf_static = None
        clf_seq = None

# --- Variables de Control y Memoria del Pipeline ---
cap = None
buffer = utils.SequenceBuffer()
prev_feats = None
motion_hist = deque(maxlen=15)

thread = None
thread_lock = Lock()
latest_frame = None

# Búfer de estabilización de predicciones estáticas (Sliding window)
static_predictions = deque(maxlen=10)
VOTING_SIZE = 10
REQUIRED_VOTES = 7

# Umbrales y Cooldowns
CONFIDENCE_THRESHOLD_STATIC = 0.80
CONFIDENCE_THRESHOLD_SEQ = 0.70
LETTER_COOLDOWN = 2.0

last_added_letter = None
last_letter_add_time = 0.0

# --- Hilo de Procesamiento de Cámara ---
def background_thread():
    """El cerebro de la IA. Lee la cámara, procesa con MediaPipe y clasifica en tiempo real."""
    global cap, prev_feats, motion_hist, buffer, latest_frame
    global last_added_letter, last_letter_add_time

    if clf_static is None or clf_seq is None:
        print("[ADVERTENCIA] Los modelos no están disponibles. Se mostrará solo el video.")
    
    if cap is None:
        # Abrir la cámara web por defecto de Windows
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara web de Windows.")
            return

    # Ajustar resolución por defecto para optimizar rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Hilo de cámara iniciado a 30 FPS.")
    last_hand_seen = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            socketio.sleep(0.01)
            continue
        
        # Efecto espejo en el frame para comportamiento de tótem interactivo
        frame = cv2.flip(frame, 1)
        
        frame_h, frame_w, _ = frame.shape
        roi_w = int(frame_w * 0.6)
        roi_h = int(frame_h * 0.75)
        roi_x = int((frame_w - roi_w) / 2)
        roi_y = int((frame_h - roi_h) / 2)
        
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        hand_detected_in_roi = False
        msg = ""
        feats = None
        
        if clf_static and clf_seq:
            # Procesar el ROI a través de MediaPipe y obtener predicción
            processed_roi, msg, current_feats, hand_detected_in_roi = utils.process_frame(
                roi.copy(), buffer, prev_feats, motion_hist, clf_static, clf_seq
            )
            
            # Reinsertar el ROI dibujado en el frame principal
            frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = processed_roi
            prev_feats = current_feats

            # Parsear mensaje de predicción: Ej. "[E] C (0.85)" o "[D] J (0.92)"
            pred_letter = None
            pred_conf = 0.0
            is_moving = False
            
            if msg:
                match = re.match(r'^\[([ED])\]\s+([A-Z])\s+\((0\.\d+|1\.0+)\)', msg)
                if match:
                    gesture_type = match.group(1) # 'E' (Estático) o 'D' (Dinámico)
                    pred_letter = match.group(2)
                    pred_conf = float(match.group(3))
                    is_moving = (gesture_type == 'D')

            # Manejar eventos de presencia y estabilidad de señas
            if hand_detected_in_roi:
                last_hand_seen = time.time()
                socketio.emit('hand_presence', {'detected': True})
                
                # Emitir estado de detección en tiempo real para mostrar en el feed
                socketio.emit('status_update', {
                    'letter': pred_letter if pred_letter else "",
                    'confidence': pred_conf,
                    'is_moving': is_moving,
                    'msg': msg
                })

                if is_moving:
                    # Si la mano se está moviendo, limpiamos el historial estático
                    static_predictions.clear()
                    
                    # Detección dinámica (J, Z)
                    if pred_letter and pred_conf >= CONFIDENCE_THRESHOLD_SEQ:
                        now = time.time()
                        if (now - last_letter_add_time) > LETTER_COOLDOWN:
                            if pred_letter != last_added_letter:
                                last_added_letter = pred_letter
                                last_letter_add_time = now
                                socketio.emit('career_detected', {
                                    'letter': pred_letter,
                                    'confidence': pred_conf
                                })
                else:
                    # Detección estática - Votar en ventana deslizante
                    if pred_letter and pred_conf >= CONFIDENCE_THRESHOLD_STATIC:
                        static_predictions.append(pred_letter)
                    else:
                        static_predictions.append("None")
                    
                    # Contar votos mayoritarios
                    non_none_votes = [v for v in static_predictions if v != "None"]
                    if non_none_votes:
                        from collections import Counter
                        counts = Counter(non_none_votes)
                        best_letter, count = counts.most_common(1)[0]
                        
                        # Si hay suficiente consenso (7 de 10 frames)
                        if count >= REQUIRED_VOTES:
                            now = time.time()
                            if (now - last_letter_add_time) > LETTER_COOLDOWN:
                                if best_letter != last_added_letter:
                                    last_added_letter = best_letter
                                    last_letter_add_time = now
                                    socketio.emit('career_detected', {
                                        'letter': best_letter,
                                        'confidence': pred_conf
                                    })
                                    static_predictions.clear()
            else:
                # No hay mano
                static_predictions.clear()
                socketio.emit('hand_presence', {'detected': False})
                
                # Si no hay mano por 2.0 segundos, desbloquear la letra para volver a detectarla
                if (time.time() - last_hand_seen) > 2.0:
                    last_added_letter = None
        else:
            socketio.emit('hand_presence', {'detected': False})

        # Dibujar marco Cyan/Amarillo de la región de interés (ROI)
        color = (255, 229, 0) if hand_detected_in_roi else (0, 229, 255)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), color, 2)
        
        with thread_lock:
            latest_frame = frame.copy()
        
        socketio.sleep(0.03) # ~30 FPS

# --- Emisión de Frames de Video (MJPEG) ---
def generate_frames():
    global latest_frame
    while True:
        with thread_lock:
            if latest_frame is None:
                time.sleep(0.03)
                continue
            ret, buffer_img = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer_img.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        socketio.sleep(0.03)

# --- Rutas de Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Controladores de WebSocket ---
@socketio.on('connect')
def handle_connect():
    print('[INFO] Cliente conectado al WebSocket del Tótem.')

@socketio.on('simulate_detection')
def handle_simulate_detection(data):
    """Permite al frontend simular una detección tocando los botones inferiores."""
    letter = data.get('letter', '').upper()
    if letter:
        print(f"[SIMULADO] Detección táctil de la carrera: {letter}")
        socketio.emit('career_detected', {
            'letter': letter,
            'confidence': 1.0,
            'simulated': True
        })

if __name__ == '__main__':
    load_models()
    # Iniciar el hilo de fondo para procesamiento de video e IA
    thread = socketio.start_background_task(target=background_thread)
    # Ejecutar en localhost puerto 5000 con soporte para Werkzeug en modo threading
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)