"""
config.py — Configuración del servicio de reconocimiento facial.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Rutas ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ── Cámara / Frame ────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Modelos ONNX ──────────────────────────────────────────────────────
YUNET_MODEL = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")

# ── Base de datos: Servicio_reconocimiento (única que usa Python) ────
DB_HOST     = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "Servicio_reconocimiento")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ── Umbrales de reconocimiento ────────────────────────────────────────
COSINE_THRESHOLD   = 0.40
MIN_FACE_SIZE      = 60
MIN_BRIGHTNESS     = 40
MAX_BLUR_VARIANCE  = 100.0

# ── Registro ──────────────────────────────────────────────────────────
MIN_EMBEDDINGS_REQUIRED = 70

# ── Servicio ──────────────────────────────────────────────────────────
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_EMBEDDINGS  = 80
POSICIONES_VALIDAS = {"centro", "izquierda", "derecha", "arriba", "abajo", "sonrisa"}

# ── Liveness ──────────────────────────────────────────────────────────
ANTI_SPOOF_MODEL = str(MODELS_DIR / "2.7_80x80_MiniFASNetV2.onnx")
LIVENESS_THRESHOLD = 0.5

# ── Detección de lentes ──────────────────────────────────────────────
# Habilitar/deshabilitar detección de lentes
EYEGLASS_DETECTION_ENABLED = os.getenv("EYEGLASS_DETECTION_ENABLED", "true").lower() == "true"

# Umbral para considerar que usa lentes (0.0 - 1.0)
# Valores más altos = más estricto (menos falsos positivos)
# Valores más bajos = más sensible (más falsos positivos)
EYEGLASS_THRESHOLD = float(os.getenv("EYEGLASS_THRESHOLD", "0.7"))

# Tamaño del caché de detecciones (número de imágenes almacenadas)
EYEGLASS_CACHE_SIZE = int(os.getenv("EYEGLASS_CACHE_SIZE", "100"))

# Tiempo de vida del caché en segundos
EYEGLASS_CACHE_TTL = int(os.getenv("EYEGLASS_CACHE_TTL", "5"))

# Tamaño del modelo: "small", "medium", "large"
EYEGLASS_MODEL_SIZE = os.getenv("EYEGLASS_MODEL_SIZE", "small")

# Tipo de lentes a detectar: "anyglasses", "sunglasses", "eyeglasses"
EYEGLASS_KIND = os.getenv("EYEGLASS_KIND", "anyglasses")

# Tiempo máximo de predicción en segundos (timeout)
EYEGLASS_PREDICTION_TIMEOUT = float(os.getenv("EYEGLASS_PREDICTION_TIMEOUT", "1.0"))

# ── Gestos ─────────────────────────────────────────────────────────────
GESTOS_VALIDOS = {"arriba", "abajo", "izquierda", "derecha", "sonrisa"}

# ── Configuración de logs ────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# ── Configuración de seguridad ───────────────────────────────────────
MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
BLOCK_TIME_MINUTES = int(os.getenv("BLOCK_TIME_MINUTES", "15"))

# ── Configuración de calidad de imagen ──────────────────────────────
MIN_FACE_QUALITY = float(os.getenv("MIN_FACE_QUALITY", "0.3"))
MAX_FACE_QUALITY = float(os.getenv("MAX_FACE_QUALITY", "0.9"))

# ── Configuración de caché de embeddings ────────────────────────────
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "3600"))

# ── Configuración de directorios ─────────────────────────────────────
SUSPICIOUS_DIR = BASE_DIR / "sospechosas"
TEMP_DIR = BASE_DIR / "temp"

# Crear directorios si no existen
SUSPICIOUS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ── Configuración de capturas ─────────────────────────────────────
# Habilitar/deshabilitar guardado de capturas
SAVE_CAPTURES_ENABLED = os.getenv("SAVE_CAPTURES_ENABLED", "true").lower() == "true"

# Guardar capturas de verificaciones exitosas (reconocido)
SAVE_RECONOCIDO = os.getenv("SAVE_RECONOCIDO", "true").lower() == "true"

# Directorio de capturas
CAPTURES_DIR = BASE_DIR / "capturas"

# Crear directorio si no existe
CAPTURES_DIR.mkdir(exist_ok=True)