"""
auth.py — Validación de API Key para endpoints.
"""
from fastapi import HTTPException, Request
from config import API_KEY
# from config import ALLOWED_IPS  # ← Descomentar cuando se implemente IP


def validate_api_key(request: Request):
    """
    Valida que la petición tenga el header X-API-Key correcto.
    """
    # ──────────────────────────────────────────────────────────────
    # VALIDACIÓN DE IP (OPCIONAL - Implementar en producción)
    # ──────────────────────────────────────────────────────────────
    # if ALLOWED_IPS:
    #     client_ip = request.client.host
    #     if client_ip not in ALLOWED_IPS:
    #         print(f"[AUTH] IP no autorizada: {client_ip}", flush=True)
    #         raise HTTPException(403, "IP no autorizada.")
    
    # ──────────────────────────────────────────────────────────────
    # VALIDACIÓN DE API KEY (OBLIGATORIO)
    # ──────────────────────────────────────────────────────────────
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        print("[AUTH] API Key no proporcionada", flush=True)
        raise HTTPException(401, "API Key no proporcionada.")
    
    if api_key != API_KEY:
        print("[AUTH] API Key inválida", flush=True)
        raise HTTPException(401, "API Key inválida.")
    
    return True