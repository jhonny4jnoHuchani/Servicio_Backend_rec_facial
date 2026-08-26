"""
main.py — Servicio de reconocimiento facial (FastAPI).
"""
import time
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from contextlib import asynccontextmanager

from config import (
    SERVICE_HOST, SERVICE_PORT, MIN_EMBEDDINGS_REQUIRED,
    MAX_IMAGE_BYTES, MAX_EMBEDDINGS, POSICIONES_VALIDAS
)
from face_detector import FaceDetector
from face_embedder import FaceEmbedder
from face_comparator import find_best_match
import database as db
from liveness_detector import LivenessDetector
from eyeglass_detector import EyeglassDetector
from skin_texture_detector import SkinTextureDetector


detector = None
embedder = None
liveness = None
eyeglass_detector = None
skin_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, embedder, liveness, eyeglass_detector, skin_detector
    print("[INFO] Cargando modelos ONNX...")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    liveness = LivenessDetector()
    
    eyeglass_detector = EyeglassDetector()
    skin_detector = SkinTextureDetector()
    print("[INFO] Modelos cargados correctamente.")
    yield

app = FastAPI(title="Servicio de Reconocimiento Facial", lifespan=lifespan)


# ── Endpoints ──────────────────────────────────────────────────────

@app.post("/register-persona")
def register_persona(ci: str = Form(...)):
    """Crea o busca una persona por CI. Retorna persona_id"""
    persona_id = db.get_or_create_persona(ci)
    return {"success": True, "persona_id": persona_id}


@app.post("/register")
def register(
    persona_id: int = Form(...),
    ci: str = Form(...),
    posicion: str = Form(...),
    image: UploadFile = File(...)
):
    """Registra un embedding facial para una persona."""
    start = time.time()

    # Validar persona
    persona = db.get_persona_by_id(persona_id)
    if not persona or persona["ci"] != ci:
        return {"success": False, "message": "Persona no encontrada o CI no coincide."}

    # Validar posición
    if posicion not in POSICIONES_VALIDAS:
        raise HTTPException(400, "Posición inválida.")

    # Tope máximo de embeddings
    if db.count_embeddings(persona_id) >= MAX_EMBEDDINGS:
        return {"success": False, "message": f"Ya alcanzó el máximo de {MAX_EMBEDDINGS} embeddings."}

    # Validar Content-Type
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Formato de imagen no válido. Usar JPG o PNG.")

    # Leer imagen
    contents = image.file.read()

    # Límite de tamaño
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Imagen demasiado grande. Máximo 5MB.")

    # Decodificar y validar
    try:
        pil_img = Image.open(BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(BytesIO(contents))
        if pil_img.format not in ("JPEG", "PNG"):
            raise HTTPException(400, "Formato no soportado.")
        frame = np.array(pil_img.convert("RGB"))
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "No se pudo procesar la imagen.")

    # Detectar rostro
    faces = detector.detect(frame)
    if faces is None:
        return {"success": False, "message": "No se detectó ningún rostro."}

    if len(faces) > 1:
        return {"success": False, "message": "Se detectaron múltiples rostros."}

    face_info = faces[0]

    # Verificar liveness
    live_result = liveness.predict(frame, face_info)
    if not live_result["is_real"]:
        from pathlib import Path
        from datetime import datetime
        sospechosa_dir = Path("sospechosas")
        sospechosa_dir.mkdir(exist_ok=True)
        filename = f"{sospechosa_dir}/spoof_register_{persona_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(filename, frame)

        db.save_log(persona_id, 0, "spoofing_detectado", liveness_score=live_result["score"])
        return {"success": False, "message": "Posible suplantación detectada.", "liveness_score": live_result["score"]}

    # Verificar lentes
    print("[MAIN] Chequeando lentes...", flush=True)
    has_eyeglass = eyeglass_detector.detect(frame, face_info)
    print(f"[MAIN] Resultado lentes: {has_eyeglass}", flush=True)

    if has_eyeglass:
        return {
            "match": False,
            "resultado": "desconocido",
            "message": "Por favor, quítese las gafas para la verificación.",
            "eyeglass_detected": True
        }


    # Verificar textura de piel
    print("[MAIN] Chequeando textura de piel...", flush=True)
    is_skin = skin_detector.analyze(frame, face_info)
    print(f"[MAIN] Resultado textura: {is_skin}", flush=True)

    if not is_skin:
        return {"success": False, "message": "No se detectó piel real.", "spoofing_detected": True}


    # Generar embedding
    try:
        embedding = embedder.extract(frame, face_info)
    except Exception as e:
        return {"success": False, "message": f"Error al extraer embedding: {str(e)}"}

    quality_score = round(float(face_info[14]), 3)
    db.save_embedding(persona_id, embedding, quality_score, posicion)

    total = db.count_embeddings(persona_id)
    tiempo_ms = int((time.time() - start) * 1000)

    return {
        "success": True,
        "message": "Embedding registrado.",
        "total_embeddings": total,
        "faltan": max(0, MIN_EMBEDDINGS_REQUIRED - total),
        "tiempo_proceso_ms": tiempo_ms
    }


@app.post("/verify")
def verify(
    request: Request,
    persona_id: int = Form(...),
    ci: str = Form(...),
    image: UploadFile = File(...)
):
    """Verifica un rostro contra los embeddings de una persona."""
    start = time.time()

    # Validar persona
    persona = db.get_persona_by_id(persona_id)
    if not persona or persona["ci"] != ci:
        return {"match": False, "resultado": "desconocido", "message": "Persona no encontrada o CI no coincide."}

    total = db.count_embeddings(persona_id)
    if total < MIN_EMBEDDINGS_REQUIRED:
        return {"match": False, "resultado": "desconocido", "message": f"Registro facial incompleto. Tiene {total} de {MIN_EMBEDDINGS_REQUIRED}."}

    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Formato de imagen no válido.")

    contents = image.file.read()
    try:
        pil_img = Image.open(BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(BytesIO(contents))
        if pil_img.format not in ("JPEG", "PNG"):
            raise HTTPException(400, "Formato no soportado.")
        frame = np.array(pil_img.convert("RGB"))
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "No se pudo procesar la imagen.")

    faces = detector.detect(frame)
    if faces is None:
        db.save_log(persona_id, 0, "desconocido")
        return {"match": False, "resultado": "desconocido", "message": "No se detectó rostro."}

    if len(faces) > 1:
        db.save_log(persona_id, 0, "desconocido")
        return {"match": False, "resultado": "desconocido", "message": "Múltiples rostros detectados."}

    face_info = faces[0]

    live_result = liveness.predict(frame, face_info)
    if not live_result["is_real"]:
        from pathlib import Path
        from datetime import datetime
        sospechosa_dir = Path("sospechosas")
        sospechosa_dir.mkdir(exist_ok=True)
        filename = f"{sospechosa_dir}/spoof_{persona_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(filename, frame)

        db.save_log(persona_id, 0, "spoofing_detectado", liveness_score=live_result["score"], ip_origen=request.client.host)
        return {"match": False, "resultado": "spoofing_detectado", "message": "Posible suplantación detectada.", "liveness_score": live_result["score"]}

    # Verificar lentes
    print("[MAIN] Chequeando lentes...", flush=True)
    has_eyeglass = eyeglass_detector.detect(frame, face_info)
    print(f"[MAIN] Resultado lentes: {has_eyeglass}", flush=True)

    if has_eyeglass:
        return {"success": False, "message": "Por favor, quítese las gafas para el registro.", "eyeglass_detected": True}

    # Verificar textura de piel
    print("[MAIN] Chequeando textura de piel...", flush=True)
    is_skin = skin_detector.analyze(frame, face_info)
    print(f"[MAIN] Resultado textura: {is_skin}", flush=True)

    if not is_skin:
        return {"match": False, "resultado": "spoofing_detectado", "message": "No se detectó piel real.", "spoofing_detected": True}

    try:
        embedding = embedder.extract(frame, face_info)

    
    except Exception as e:
        return {"match": False, "resultado": "desconocido", "message": str(e)}

    embeddings = db.get_embeddings_by_persona(persona_id)

    if not embeddings:
        db.save_log(persona_id, 0, "desconocido")
        return {"match": False, "resultado": "desconocido", "message": "No hay embeddings registrados."}

    result = find_best_match(embedding, embeddings)
    tiempo_ms = int((time.time() - start) * 1000)
    resultado = "reconocido" if result["match"] else "desconocido"

    db.save_log(persona_id, result["confidence"], resultado, tiempo_proceso_ms=tiempo_ms, ip_origen=request.client.host)

    return {
        "match": result["match"],
        "resultado": resultado,
        "confianza": result["confidence"],
        "tiempo_proceso_ms": tiempo_ms
    }


@app.get("/status/{persona_id}")
def status(persona_id: int):
    """Estado del registro facial de una persona."""
    info = db.get_reconocimiento_status(persona_id)
    return {
        "persona_id": persona_id,
        "activo": info["activo"],
        "total_embeddings": info["total_embeddings"],
        "calidad_promedio": info["calidad_promedio"],
        "habilitado": info["total_embeddings"] >= MIN_EMBEDDINGS_REQUIRED
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "reconocimiento-facial"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)