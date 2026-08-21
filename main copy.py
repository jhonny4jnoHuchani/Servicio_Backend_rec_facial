"""
main.py — Servicio de reconocimiento facial (FastAPI).
"""
import time
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Request
from contextlib import asynccontextmanager

from config import (SERVICE_HOST, SERVICE_PORT, MIN_EMBEDDINGS_REQUIRED,MAX_IMAGE_BYTES, MAX_EMBEDDINGS, POSICIONES_VALIDAS)
from face_detector import FaceDetector
from face_embedder import FaceEmbedder
from face_comparator import find_best_match
import database as db
from liveness_detector import LivenessDetector



# Carga global de modelos (una sola vez al iniciar)
detector = None
embedder = None
liveness = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, embedder, liveness
    print("[INFO] Cargando modelos ONNX...")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    liveness = LivenessDetector()
    print("[INFO] Modelos cargados correctamente.")
    yield

app = FastAPI(title="Servicio de Reconocimiento Facial", lifespan=lifespan)


# ── Endpoints ──────────────────────────────────────────────────────

@app.post("/register")
def register(
    docente_id: int = Form(...),
    posicion: str = Form(...),
    image: UploadFile = File(...)
):
    """Registra un embedding facial para un docente."""
    start = time.time()

    # Validar posición
    if posicion not in POSICIONES_VALIDAS:
        raise HTTPException(400, "Posición inválida.")

    # Tope máximo de embeddings
    
    if db.count_embeddings(docente_id) >= MAX_EMBEDDINGS:
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
            raise HTTPException(400, "Formato no soportado. Use JPG o PNG.")
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

    live_result = liveness.predict(frame, face_info)
    if not live_result["is_real"]:
        # Guardar imagen sospechosa
        from pathlib import Path
        from datetime import datetime
        sospechosa_dir = Path("sospechosas")
        sospechosa_dir.mkdir(exist_ok=True)
        filename = f"{sospechosa_dir}/spoof_register_{docente_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(filename, frame)

        db.save_log(docente_id, 0, "spoofing_detectado", liveness_score=live_result["score"])
        
        return {"success": False, "message": "Posible suplantación detectada.", "liveness_score": live_result["score"]}
    # Generar embedding
    try:
        embedding = embedder.extract(frame, face_info)
    except Exception as e:
        return {"success": False, "message": f"Error al extraer embedding: {str(e)}"}

    # Obtener o crear registro en reconocimiento_facial
    rec_id = db.get_or_create_reconocimiento(docente_id)

    # Calcular quality_score
    quality_score = round(float(face_info[14]), 3)

    # Guardar embedding
    db.save_embedding(docente_id, rec_id, embedding, quality_score, posicion)

    # Actualizar estadísticas
    db.update_reconocimiento_stats(docente_id)

    total = db.count_embeddings(docente_id)
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
    docente_id: int = Form(...),
    image: UploadFile = File(...)
):
    """Verifica un rostro contra los embeddings del docente."""
    start = time.time()
    
    # Verificar si tiene suficientes embeddings
    total = db.count_embeddings(docente_id)
    if total < MIN_EMBEDDINGS_REQUIRED:
        return {
            "match": False,
            "resultado": "desconocido",
            "message": f"Registro facial incompleto. Tiene {total} de {MIN_EMBEDDINGS_REQUIRED} requeridos."
        }

    # Leer imagen
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Formato de imagen no válido.")

    contents = image.file.read()
    try:
        pil_img = Image.open(BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(BytesIO(contents))
        if pil_img.format not in ("JPEG", "PNG"):
            raise HTTPException(400, "Formato no soportado. Use JPG o PNG.")
        frame = np.array(pil_img.convert("RGB"))
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "No se pudo procesar la imagen.")

    # Detectar rostro
    faces = detector.detect(frame)
    if faces is None:
        db.save_log(docente_id, 0, "desconocido")
        return {"match": False, "resultado": "desconocido", "message": "No se detectó rostro."}

    if len(faces) > 1:
        db.save_log(docente_id, 0, "desconocido")
        return {"match": False, "resultado": "desconocido", "message": "Múltiples rostros detectados."}

    face_info = faces[0]
    # Verificar liveness
    live_result = liveness.predict(frame, face_info)

    if not live_result["is_real"]:
        # Guardar imagen sospechosa
        from pathlib import Path
        from datetime import datetime
        sospechosa_dir = Path("sospechosas")
        sospechosa_dir.mkdir(exist_ok=True)
        filename = f"{sospechosa_dir}/spoof_{docente_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(filename, frame)
        
        db.save_log(docente_id, 0, "spoofing_detectado", liveness_score=live_result["score"], ip_origen=request.client.host)
        return {"match": False, "resultado": "spoofing_detectado", "message": "Posible suplantación detectada.", "liveness_score": live_result["score"]}


    # Generar embedding
    try:
        embedding = embedder.extract(frame, face_info)
    except Exception as e:
        return {"match": False, "resultado": "desconocido", "message": str(e)}

    # Cargar embeddings del docente
    embeddings = db.get_embeddings_by_docente(docente_id)

    if not embeddings:
        db.save_log(docente_id, 0, "desconocido")
        return {"match": False, "resultado": "desconocido", "message": "No hay embeddings registrados."}

    # Comparar
    result = find_best_match(embedding, embeddings)
    tiempo_ms = int((time.time() - start) * 1000)

    # Guardar log
    resultado = "reconocido" if result["match"] else "desconocido"
    db.save_log(docente_id, result["confidence"], resultado, tiempo_proceso_ms=tiempo_ms, ip_origen=request.client.host)

    return {
        "match": result["match"],
        "resultado": resultado,
        "confianza": result["confidence"],
        "tiempo_proceso_ms": tiempo_ms
    }


@app.get("/status/{docente_id}")
def status(docente_id: int):
    """Estado del registro facial de un docente."""
    info = db.get_reconocimiento_status(docente_id)
    return {
        "docente_id": docente_id,
        "activo": info["activo"],
        "total_embeddings": info["total_embeddings"],
        "calidad_promedio": info["calidad_promedio"],
        "habilitado": info["total_embeddings"] >= MIN_EMBEDDINGS_REQUIRED
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "reconocimiento-facial"}


# ── Inicio ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)