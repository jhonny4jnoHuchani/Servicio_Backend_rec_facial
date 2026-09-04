"""
logger.py — Logger para verificaciones.
"""
import logging
from datetime import datetime
from pathlib import Path

# Crear carpeta logs
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Logger para verify
verify_logger = logging.getLogger('verify')
verify_logger.setLevel(logging.INFO)

# Evitar duplicados si se importa múltiples veces
if not verify_logger.handlers:
    # Archivo por día
    log_file = LOG_DIR / f"verify-{datetime.now():%Y-%m-%d}.log"
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    verify_logger.addHandler(handler)
    
    # También mostrar en consola
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    verify_logger.addHandler(console)