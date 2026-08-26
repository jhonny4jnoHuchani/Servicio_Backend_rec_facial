"""
eyeglass_detector.py — Detección de lentes con glasses-detector.
"""
import numpy as np
import threading
from config import EYEGLASS_THRESHOLD

class EyeglassDetector:
    """
    Usa el clasificador preentrenado glasses-detector para
    detectar si la persona tiene lentes puestos.
    """

    def __init__(self):
        from glasses_detector import GlassesClassifier
        self.classifier = GlassesClassifier(size="small", kind="anyglasses")
        self._lock = threading.Lock()

    def detect(self, frame: np.ndarray, face_info: np.ndarray) -> bool:
        print("[EYEGLASS] detect() llamado", flush=True)
        x, y, w, h = face_info[:4].astype(int)
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            print("[EYEGLASS] Crop vacío", flush=True)
            return False

        rgb_crop = face_crop[:, :, ::-1]

        try:
            with self._lock:
                proba = self.classifier(rgb_crop, format="proba")
            print(f"[EYEGLASS] proba={proba:.4f}", flush=True)
            return proba > EYEGLASS_THRESHOLD
        except Exception as e:
            print(f"[EYEGLASS] Error: {e}", flush=True)
            return False