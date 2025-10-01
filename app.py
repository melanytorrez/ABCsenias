# app.py (Versión Corregida para Carga de Modelos)

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import time
from collections import deque
import lsb_mvp_utils as utils
from threading import Lock
import joblib # <<--- Se añade joblib aquí
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# --- Configuración de Modelos y Carga (Sólo aquí) ---
MODEL_PATH = Path("models/lsb_alpha.joblib")
MODEL_SEQ_PATH = Path("models/lsb_seq.joblib")

# Variables para almacenar los modelos cargados
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
        # Esto evitará que la app falle, pero la predicción estará vacía
        clf_static = None
        clf_seq = None
        # Puedes salir aquí si es crítico
        # sys.exit(1)


# --- Variables Globales y Lógica ---
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


def background_thread():
    """El 'cerebro' de la aplicación. Se ejecuta en segundo plano."""
    global cap, prev_feats, motion_hist, buffer, latest_frame
    global current_sentence, last_added_letter, last_hand_seen_time, last_letter_add_time
    
    # Comprobar si los modelos fueron cargados (por si solo se hizo 'collect')
    if clf_static is None or clf_seq is None:
        print("[ADVERTENCIA] Modelos no disponibles. La aplicación de web solo mostrará el video.")
    
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
        
        # <<--- CAMBIO CLAVE: Pasamos los modelos como argumentos ---
        if clf_static and clf_seq:
            processed_frame, msg, current_feats, hand_detected = utils.process_frame(
                frame.copy(), buffer, prev_feats, motion_hist, clf_static, clf_seq
            )
            prev_feats = current_feats

            # Lógica de escritura (solo si hay modelos)
            if hand_detected:
                last_hand_seen_time = time.time()
                
                if (time.time() - last_letter_add_time) > LETTER_COOLDOWN:
                    if msg and len(msg.split(' ')) > 1:
                        predicted_letter = msg.split(' ')[1]
                        if predicted_letter != last_added_letter:
                            current_sentence.append(predicted_letter)
                            last_added_letter = predicted_letter
                            last_letter_add_time = time.time()
                            socketio.emit('update_text', {'text': "".join(current_sentence)})
            else:
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
            # Si los modelos no cargaron, mostramos el frame sin dibujar la predicción
            processed_frame = frame
        
        with thread_lock:
            latest_frame = processed_frame.copy()
        
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

if __name__ == '__main__':
    # Llamamos a la carga de modelos antes de ejecutar la aplicación
    load_models() 
    socketio.run(app, host='0.0.0.0', debug=True)