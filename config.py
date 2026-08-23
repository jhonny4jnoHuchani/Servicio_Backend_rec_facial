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
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")

# ── Umbrales de reconocimiento ────────────────────────────────────────
COSINE_THRESHOLD   = 0.40
MIN_FACE_SIZE      = 60
MIN_BRIGHTNESS     = 40
MAX_BLUR_VARIANCE  = 100.0

# ── Registro ──────────────────────────────────────────────────────────
MIN_EMBEDDINGS_REQUIRED = 50

# ── Servicio ──────────────────────────────────────────────────────────
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_EMBEDDINGS  = 60
POSICIONES_VALIDAS = {"centro", "izquierda", "derecha", "arriba", "abajo", "sonrisa"}

# ── Liveness ──────────────────────────────────────────────────────────
ANTI_SPOOF_MODEL = str(MODELS_DIR / "2.7_80x80_MiniFASNetV2.onnx")
LIVENESS_THRESHOLD = 0.5