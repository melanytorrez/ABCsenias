# app.py (Versión Final Definitiva con Inicialización Perezosa de la Cámara)

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import time
from collections import deque
import lsb_mvp_utils as utils
from threading import Lock

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# --- Variables Globales y Sincronización ---
# <<--- CAMBIO CLAVE: No inicializamos la cámara aquí.
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
SPACE_TIMEOUT = 4.0

def background_thread():
    """El 'cerebro' de la aplicación. Se ejecuta en segundo plano."""
    global cap, prev_feats, motion_hist, buffer, latest_frame
    global current_sentence, last_added_letter, last_hand_seen_time
    
    # <<--- CAMBIO CLAVE: Inicializamos la cámara aquí, dentro del hilo.
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara. El hilo de fondo no se puede iniciar.")
            return # Detiene la ejecución del hilo si la cámara falla.

    print("Iniciando hilo de fondo...")
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        
        processed_frame, msg, current_feats, hand_detected = utils.process_frame(
            frame.copy(), buffer, prev_feats, motion_hist
        )
        prev_feats = current_feats

        if hand_detected:
            last_hand_seen_time = time.time()
            if msg and len(msg.split(' ')) > 1:
                predicted_letter = msg.split(' ')[1]
                if predicted_letter != last_added_letter:
                    current_sentence.append(predicted_letter)
                    last_added_letter = predicted_letter
                    socketio.emit('update_text', {'text': "".join(current_sentence)})
        else:
            if (time.time() - last_hand_seen_time) > SPACE_TIMEOUT:
                if current_sentence and current_sentence[-1] != ' ':
                    current_sentence.append(' ')
                    last_added_letter = ' '
                    socketio.emit('update_text', {'text': "".join(current_sentence)})
                    last_hand_seen_time = time.time()
            if last_added_letter is not None and last_added_letter != ' ':
                last_added_letter = None
        
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
    socketio.run(app, host='0.0.0.0', debug=True)