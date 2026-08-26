"""
liveness_detector.py — Detección de anti-spoofing con MiniFASNet.
"""
import cv2
import numpy as np
import threading
from pathlib import Path
from config import ANTI_SPOOF_MODEL, LIVENESS_THRESHOLD

REAL_CLASS_INDEX = 2  # Índice de "real" en las 3 clases del modelo


class LivenessDetector:
    def __init__(self):
        if not Path(ANTI_SPOOF_MODEL).exists():
            raise FileNotFoundError(f"Modelo anti-spoofing no encontrado: {ANTI_SPOOF_MODEL}")
        self._model = cv2.dnn.readNetFromONNX(ANTI_SPOOF_MODEL)
        self._lock = threading.Lock()

    def _crop_with_scale(self, frame, face_info, scale=2.7, out_size=80):
        """Recorta el rostro con un factor de escala 2.7x, centrado."""
        x, y, w, h = face_info[:4]
        cx, cy = x + w / 2, y + h / 2
        new_size = max(w, h) * scale
        half = new_size / 2

        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(frame.shape[1], cx + half))
        y2 = int(min(frame.shape[0], cy + half))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (out_size, out_size))

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict(self, frame, face_info):
        print("[LIVENESS] Iniciando análisis...", flush=True)
        
        face = self._crop_with_scale(frame, face_info)
        if face is None:
            print("[LIVENESS] No se pudo recortar rostro", flush=True)
            return {"is_real": False, "score": 0.0}

        face = face.astype(np.float32) / 255.0
        face = face.transpose(2, 0, 1)
        face = np.expand_dims(face, axis=0)

        with self._lock:
            self._model.setInput(face)
            output = self._model.forward()

        probs = self._softmax(output[0])
        score = float(probs[REAL_CLASS_INDEX])
        
        print(f"[LIVENESS] Score real={score:.4f} | Fake1={probs[0]:.4f} | Fake2={probs[1]:.4f} | Umbral={LIVENESS_THRESHOLD}", flush=True)
        print(f"[LIVENESS] Resultado: {'✅ REAL' if score >= LIVENESS_THRESHOLD else '❌ SPOOFING (foto/pantalla)'}", flush=True)

        return {
            "is_real": score >= LIVENESS_THRESHOLD,
            "score": round(score, 4)
        }