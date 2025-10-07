# app.py (Versión con cuadro de reconocimiento y función para BORRAR)

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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# --- Configuración de Modelos y Carga (Sin cambios) ---
MODEL_PATH = Path("models/lsb_alpha.joblib")
MODEL_SEQ_PATH = Path("models/lsb_seq.joblib")

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
        print("[INFO] Modelos de IA cargados correctamente.")
    except FileNotFoundError:
        print("\n[ERROR] No se pudieron cargar los modelos.")
        print("Asegúrate de haber ejecutado 'python lsb_mvp.py train' y 'python lsb_mvp.py train-seq'.")
        clf_static = None
        clf_seq = None

# --- Variables Globales y Lógica (Sin cambios) ---
cap = None
buffer = utils.SequenceBuffer()
prev_feats = None
motion_hist = deque(maxlen=15)

thread = None
thread_lock = Lock()
latest_frame = None

current_sentence = []
last_added_letter = None
last_hand_seen_time = time.time()
last_letter_add_time = 0
SPACE_TIMEOUT = 4.0
LETTER_COOLDOWN = 2.0

hand_in_roi_start_time = None
RECOGNITION_DELAY_SECONDS = 1.0
roi_color = (255, 255, 0)

def background_thread():
    """El 'cerebro' de la aplicación. Se ejecuta en segundo plano."""
    global cap, prev_feats, motion_hist, buffer, latest_frame
    global current_sentence, last_added_letter, last_hand_seen_time, last_letter_add_time
    global hand_in_roi_start_time, roi_color

    if clf_static is None or clf_seq is None:
        print("[ADVERTENCIA] Modelos no disponibles. La aplicación web solo mostrará el video.")
    
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

    print("Iniciando hilo de fondo...")
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        
        frame_h, frame_w, _ = frame.shape
        roi_w = int(frame_w * 0.6)
        roi_h = int(frame_h * 0.75)
        roi_x = int((frame_w - roi_w) / 2)
        roi_y = int((frame_h - roi_h) / 2)
        
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        hand_detected_in_roi = False
        
        if clf_static and clf_seq:
            processed_roi, msg, current_feats, hand_detected_in_roi = utils.process_frame(
                roi.copy(), buffer, prev_feats, motion_hist, clf_static, clf_seq
            )
            
            frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = processed_roi
            prev_feats = current_feats

            if hand_detected_in_roi:
                last_hand_seen_time = time.time()
                
                if hand_in_roi_start_time is None:
                    hand_in_roi_start_time = time.time()
                    roi_color = (0, 255, 255)
                else:
                    elapsed_time = time.time() - hand_in_roi_start_time
                    
                    if elapsed_time >= RECOGNITION_DELAY_SECONDS:
                        roi_color = (0, 255, 0)
                        
                        if (time.time() - last_letter_add_time) > LETTER_COOLDOWN:
                            if msg and len(msg.split(' ')) > 1:
                                predicted_letter = msg.split(' ')[1]
                                if predicted_letter != last_added_letter:
                                    current_sentence.append(predicted_letter)
                                    last_added_letter = predicted_letter
                                    last_letter_add_time = time.time()
                                    socketio.emit('update_text', {'text': "".join(current_sentence)})
                        hand_in_roi_start_time = None 
            else:
                hand_in_roi_start_time = None
                roi_color = (255, 255, 0)
                
                if (time.time() - last_hand_seen_time) > SPACE_TIMEOUT:
                    if current_sentence and current_sentence[-1] != ' ':
                        current_sentence.append(' ')
                        last_added_letter = ' '
                        socketio.emit('update_text', {'text': "".join(current_sentence)})
                        last_hand_seen_time = time.time()
                        last_letter_add_time = 0
                if last_added_letter is not None and last_added_letter != ' ':
                    last_added_letter = None
        else:
            processed_frame = frame
        
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), roi_color, 2)
        
        status_text = "Coloque la mano aqui"
        if hand_in_roi_start_time is not None:
            status_text = "Estabilice la mano..."
            if roi_color == (0, 255, 0): status_text = "¡Reconocido!"

        cv2.putText(frame, status_text, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)
        
        with thread_lock:
            latest_frame = frame.copy()
        
        socketio.sleep(0.05)

def generate_frames():
    """Los 'ojos' de la aplicación. Solo sirve el video."""
    global latest_frame
    while True:
        with thread_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer_img = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer_img.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        socketio.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    global thread
    print('Cliente conectado')
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_thread)
    socketio.emit('update_text', {'text': "".join(current_sentence)})

# --- NUEVA FUNCIÓN PARA BORRAR EL TEXTO ---
@socketio.on('clear_text')
def handle_clear_text():
    """Limpia la oración actual cuando recibe la señal del cliente."""
    global current_sentence, last_added_letter
    current_sentence = []
    last_added_letter = None
    # Informa a todos los clientes que el texto ahora está vacío
    socketio.emit('update_text', {'text': ""})
    print('[INFO] Texto borrado por el cliente.')
# --- FIN DE LA NUEVA FUNCIÓN ---

if __name__ == '__main__':
    load_models() 
    socketio.run(app, host='0.0.0.0', debug=True)