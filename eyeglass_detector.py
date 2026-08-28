"""
eyeglass_detector.py — Detección de lentes con glasses-detector.
"""
import numpy as np
import threading
import time
from typing import Optional, Tuple
from config import (
    EYEGLASS_THRESHOLD,
    EYEGLASS_DETECTION_ENABLED,
    EYEGLASS_CACHE_SIZE,
    EYEGLASS_CACHE_TTL,
    EYEGLASS_MODEL_SIZE,
    EYEGLASS_KIND,
    EYEGLASS_PREDICTION_TIMEOUT
)

class EyeglassDetector:
    """
    Usa el clasificador preentrenado glasses-detector para
    detectar si la persona tiene lentes puestos.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._classifier = None
        self._cache = {}
        self._cache_max_size = EYEGLASS_CACHE_SIZE
        self._cache_ttl = EYEGLASS_CACHE_TTL
        self._initialized = False
        self._prediction_timeout = EYEGLASS_PREDICTION_TIMEOUT
        self._stats = {
            "total_detections": 0,
            "positive_detections": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
        self._initialize_classifier()

    def _initialize_classifier(self):
        """Inicializa el clasificador de lentes"""
        try:
            from glasses_detector import GlassesClassifier
            self._classifier = GlassesClassifier(
                size=EYEGLASS_MODEL_SIZE,
                kind=EYEGLASS_KIND
            )
            self._initialized = True
            print(f"[EYEGLASS] Clasificador inicializado (size={EYEGLASS_MODEL_SIZE}, kind={EYEGLASS_KIND})", flush=True)
        except ImportError as e:
            print(f"[EYEGLASS] Error importando glasses_detector: {e}", flush=True)
            self._initialized = False
        except Exception as e:
            print(f"[EYEGLASS] Error inicializando clasificador: {e}", flush=True)
            self._initialized = False

    def detect(self, frame: np.ndarray, face_info: np.ndarray) -> bool:
        """
        Detecta si la persona usa lentes.
        
        Args:
            frame: Imagen completa (BGR)
            face_info: Información del rostro [x, y, w, h, ...]
        
        Returns:
            bool: True si usa lentes, False en caso contrario
        """
        # Si la detección está deshabilitada
        if not EYEGLASS_DETECTION_ENABLED:
            print("[EYEGLASS] Detección deshabilitada", flush=True)
            return False

        if not self._initialized or self._classifier is None:
            print("[EYEGLASS] Clasificador no disponible", flush=True)
            return False

        print("[EYEGLASS] detect() llamado", flush=True)

        try:
            # Validar entrada
            if frame is None or frame.size == 0:
                print("[EYEGLASS] Frame vacío", flush=True)
                return False

            if face_info is None or len(face_info) < 4:
                print("[EYEGLASS] Face_info inválido", flush=True)
                return False

            # Extraer coordenadas del rostro
            x, y, w, h = face_info[:4].astype(int)
            
            # Validar coordenadas
            if w <= 0 or h <= 0:
                print("[EYEGLASS] Dimensiones inválidas", flush=True)
                return False

            # Asegurar que las coordenadas estén dentro de la imagen
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)

            if w <= 0 or h <= 0:
                print("[EYEGLASS] Dimensiones inválidas después de ajuste", flush=True)
                return False

            # Recortar la región del rostro
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                print("[EYEGLASS] Crop vacío", flush=True)
                return False

            # Generar clave de caché
            cache_key = self._generate_cache_key(face_crop)

            # Verificar caché
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self._stats["cache_hits"] += 1
                print(f"[EYEGLASS] Resultado desde caché: {cached_result}", flush=True)
                return cached_result

            self._stats["cache_misses"] += 1

            # Convertir BGR a RGB
            rgb_crop = face_crop[:, :, ::-1]

            # Ejecutar clasificador con timeout
            proba = self._predict_with_timeout(rgb_crop)
            
            if proba is None:
                print("[EYEGLASS] Error en predicción", flush=True)
                self._stats["errors"] += 1
                return False

            result = proba > EYEGLASS_THRESHOLD
            
            # Actualizar estadísticas
            self._stats["total_detections"] += 1
            if result:
                self._stats["positive_detections"] += 1
            
            # Guardar en caché
            self._add_to_cache(cache_key, result)
            
            print(f"[EYEGLASS] proba={proba:.4f}, threshold={EYEGLASS_THRESHOLD}, result={result}", flush=True)
            return result

        except Exception as e:
            print(f"[EYEGLASS] Error en detect: {e}", flush=True)
            self._stats["errors"] += 1
            return False

    def _predict_with_timeout(self, rgb_crop: np.ndarray) -> Optional[float]:
        """
        Ejecuta la predicción con timeout.
        
        Args:
            rgb_crop: Imagen en formato RGB
        
        Returns:
            float: Probabilidad de usar lentes, o None si hay error
        """
        result = [None]
        exception = [None]

        def run_prediction():
            try:
                with self._lock:
                    result[0] = self._classifier(rgb_crop, format="proba")
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_prediction)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self._prediction_timeout)

        if thread.is_alive():
            print(f"[EYEGLASS] Timeout en predicción ({self._prediction_timeout}s)", flush=True)
            return None

        if exception[0] is not None:
            print(f"[EYEGLASS] Error en predicción: {exception[0]}", flush=True)
            return None

        return result[0]

    def _generate_cache_key(self, image: np.ndarray) -> str:
        """
        Genera una clave de caché a partir de la imagen.
        
        Args:
            image: Imagen numpy array
        
        Returns:
            str: Clave de caché
        """
        # Redimensionar para consistencia (reducir ruido)
        step = max(1, min(image.shape[0], image.shape[1]) // 10)
        small = image[::step, ::step]
        hash_value = hash(small.tobytes())
        return f"eyeglass_{hash_value}"

    def _get_from_cache(self, key: str) -> Optional[bool]:
        """
        Obtiene un resultado del caché si es válido.
        
        Args:
            key: Clave de caché
        
        Returns:
            bool o None si no está en caché o expiró
        """
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            else:
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, value: bool):
        """
        Agrega un resultado al caché.
        
        Args:
            key: Clave de caché
            value: Resultado a almacenar
        """
        if len(self._cache) >= self._cache_max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[key] = (value, time.time())

    def clear_cache(self):
        """Limpia el caché de detecciones"""
        self._cache.clear()
        print("[EYEGLASS] Caché limpiado", flush=True)

    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del detector.
        
        Returns:
            dict: Estadísticas
        """
        return {
            "initialized": self._initialized,
            "cache_size": len(self._cache),
            "cache_max_size": self._cache_max_size,
            "threshold": EYEGLASS_THRESHOLD,
            "enabled": EYEGLASS_DETECTION_ENABLED,
            "model_size": EYEGLASS_MODEL_SIZE,
            "kind": EYEGLASS_KIND,
            "prediction_timeout": self._prediction_timeout,
            "total_detections": self._stats["total_detections"],
            "positive_detections": self._stats["positive_detections"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "errors": self._stats["errors"]
        }

    def detect_batch(self, frame: np.ndarray, faces_info: list) -> list:
        """
        Detecta lentes en múltiples rostros.
        
        Args:
            frame: Imagen completa
            faces_info: Lista de información de rostros
        
        Returns:
            list: Lista de booleanos
        """
        results = []
        for face_info in faces_info:
            result = self.detect(frame, face_info)
            results.append(result)
        return results

    def detect_with_confidence(self, frame: np.ndarray, face_info: np.ndarray) -> Tuple[bool, float]:
        """
        Detecta lentes y devuelve la confianza.
        
        Args:
            frame: Imagen completa
            face_info: Información del rostro
        
        Returns:
            Tuple[bool, float]: (resultado, confianza)
        """
        try:
            if not EYEGLASS_DETECTION_ENABLED:
                return False, 0.0

            if not self._initialized or self._classifier is None:
                return False, 0.0

            x, y, w, h = face_info[:4].astype(int)
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                return False, 0.0

            rgb_crop = face_crop[:, :, ::-1]
            
            with self._lock:
                proba = self._classifier(rgb_crop, format="proba")
            
            return proba > EYEGLASS_THRESHOLD, float(proba)
            
        except Exception as e:
            print(f"[EYEGLASS] Error en detect_with_confidence: {e}", flush=True)
            return False, 0.0