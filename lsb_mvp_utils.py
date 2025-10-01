# lsb_mvp_utils.py (Versión Corregida sin Carga de Modelos)

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pathlib import Path

# --- Configuración Base ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
SEQ_LEN = 20

# --- Clases y Funciones de Procesamiento ---
class SequenceBuffer:
    def __init__(self, maxlen=SEQ_LEN):
        self.frames = deque(maxlen=maxlen)
    def add(self, feats):
        if feats is not None: self.frames.append(feats)
    def is_ready(self):
        return len(self.frames) == self.frames.maxlen
    def get_sequence(self):
        return np.concatenate(self.frames)

def landmarks_to_features(landmarks):
    if landmarks is None or len(landmarks) != 21: return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts_rel = pts - wrist
    dists = [np.linalg.norm(pts[i] - wrist) for i in [5, 9, 13, 17]]
    hand_size = (np.mean(dists) + 1e-6)
    pts_rel /= hand_size
    return pts_rel.reshape(-1)

def motion_score(prev_feats, curr_feats):
    if prev_feats is None or curr_feats is None: return 0.0
    return np.linalg.norm(curr_feats - prev_feats)

# --- Función Principal de Reconocimiento ---
# <<--- CAMBIO CLAVE: Recibe clf_static y clf_seq como parámetros
def process_frame(frame, buffer, prev_feats, motion_hist, clf_static, clf_seq):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    msg = ""
    feats = None
    hand_detected = False

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            hand_detected = True
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            feats = landmarks_to_features(res.multi_hand_landmarks[0].landmark)
            
            if prev_feats is not None:
                score = motion_score(prev_feats, feats)
                motion_hist.append(score)

            moving = sum(s > 0.15 for s in motion_hist)
            if moving >= 8:
                buffer.add(feats)
                msg = "[DINAMICO...]"
                if buffer.is_ready():
                    X = buffer.get_sequence().reshape(1, -1)
                    proba = clf_seq.predict_proba(X)[0]
                    conf = np.max(proba)
                    if conf >= 0.7:
                        pred = clf_seq.classes_[np.argmax(proba)]
                        msg = f"[D] {pred} ({conf:.2f})"
            else:
                X = feats.reshape(1, -1)
                proba = clf_static.predict_proba(X)[0]
                pred = clf_static.classes_[np.argmax(proba)]
                conf = np.max(proba)
                msg = f"[E] {pred} ({conf:.2f})"
    
    return frame, msg, feats, hand_detected