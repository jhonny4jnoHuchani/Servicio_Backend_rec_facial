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

from gesture_detector import GestureDetector
from config import GESTOS_VALIDOS
from capture_manager import capture_manager

detector = None
embedder = None
liveness = None
eyeglass_detector = None
gesture_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, embedder, liveness, eyeglass_detector, gesture_detector
    print("[INFO] Cargando modelos ONNX...")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    liveness = LivenessDetector()
    gesture_detector = GestureDetector()
    eyeglass_detector = EyeglassDetector()
    
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
        print(f"[REGISTER] Falló: no detectó rostro (persona_id={persona_id})", flush=True)
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
    gesto_solicitado: str = Form(...),
    foto_frontal: UploadFile = File(...),
    foto_gesto: UploadFile = File(...),
    tipo_marcado: str = Form("verificando")
):
    """Verifica un rostro contra los embeddings de una persona."""
    start = time.time()

    # ══════════════════════════════════════════════════════════════
    # PASO 1: Validar persona y CI
    # ══════════════════════════════════════════════════════════════
    persona = db.get_persona_by_id(persona_id)
    if not persona or persona["ci"] != ci:
        print(f"[VERIFY] Persona no encontrada o CI no coincide (persona_id={persona_id})", flush=True)
        return {"match": False, "resultado": "desconocido", "message": "Persona no encontrada o CI no coincide."}

    # ══════════════════════════════════════════════════════════════
    # PASO 2: Validar embeddings mínimos
    # ══════════════════════════════════════════════════════════════
    total = db.count_embeddings(persona_id)
    if total < MIN_EMBEDDINGS_REQUIRED:
        print(f"[VERIFY] Registro incompleto: {total}/{MIN_EMBEDDINGS_REQUIRED}", flush=True)
        return {"match": False, "resultado": "desconocido", "message": f"Registro facial incompleto. Tiene {total} de {MIN_EMBEDDINGS_REQUIRED}."}

    # ══════════════════════════════════════════════════════════════
    # PASO 3: Validar gesto solicitado
    # ══════════════════════════════════════════════════════════════
    if gesto_solicitado not in GESTOS_VALIDOS:
        print(f"[VERIFY] Gesto solicitado inválido: {gesto_solicitado}", flush=True)
        return {"match": False, "resultado": "desconocido", "message": "Gesto no válido."}

    # ══════════════════════════════════════════════════════════════
    # PASO 4: Procesar foto_frontal
    # ══════════════════════════════════════════════════════════════
    if foto_frontal.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Formato de imagen no válido.")

    contents = foto_frontal.file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Imagen demasiado grande. Máximo 5MB.")

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

    # ══════════════════════════════════════════════════════════════
    # PASO 5: Detectar rostro en foto_frontal
    # ══════════════════════════════════════════════════════════════
    faces = detector.detect(frame)

    if faces is None:
        print("[VERIFY] No se detectó rostro en foto frontal", flush=True)
        ruta_img = capture_manager.save(frame, ci, "error", tipo_marcado)
        db.save_log(persona_id, 0, "desconocido", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {"match": False, "resultado": "desconocido", "message": "No se detectó rostro."}


    if len(faces) > 1:
        print(f"[VERIFY] Múltiples rostros en foto frontal: {len(faces)}", flush=True)
        ruta_img = capture_manager.save(frame, ci, "error", tipo_marcado)
        db.save_log(persona_id, 0, "desconocido", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {"match": False, "resultado": "desconocido", "message": "Múltiples rostros detectados."}


    face_info = faces[0]

    # ══════════════════════════════════════════════════════════════
    # PASO 6: Verificar LENTES en foto_frontal
    # ══════════════════════════════════════════════════════════════
    print("[VERIFY] Paso 1: Verificando lentes en foto frontal...", flush=True)
    has_eyeglass = eyeglass_detector.detect(frame, face_info)
    print(f"[VERIFY] Resultado lentes frontal: {has_eyeglass}", flush=True)

    if has_eyeglass:
        ruta_img = capture_manager.save(frame, ci, "con_lentes", tipo_marcado)
        db.save_log(persona_id, 0, "con_lentes", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {
            "match": False,
            "resultado": "desconocido",
            "message": "Por favor, quítese las gafas para la verificación.",
            "eyeglass_detected": True
        }

    # ══════════════════════════════════════════════════════════════
    # PASO 7: Verificar LIVENESS en foto_frontal
    # ══════════════════════════════════════════════════════════════
    print("[VERIFY] Paso 2: Verificando liveness en foto frontal...", flush=True)
    live_result = liveness.predict(frame, face_info)
    print(f"[VERIFY] Resultado liveness frontal: is_real={live_result['is_real']} score={live_result['score']:.4f}", flush=True)


    if not live_result["is_real"]:
        ruta_img = capture_manager.save(frame, ci, "spoofing_detectado", tipo_marcado)
        db.save_log(persona_id, 0, "spoofing_detectado",
                liveness_score=live_result["score"],
                ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {
            "match": False,
            "resultado": "spoofing_detectado",
            "message": "Posible suplantación detectada.",
            "liveness_score": live_result["score"]
        }
    

    # ══════════════════════════════════════════════════════════════
    # PASO 8: Generar embedding + COMPARAR identidad
    # ══════════════════════════════════════════════════════════════
    print("[VERIFY] Paso 3: Generando embedding y comparando identidad...", flush=True)
    try:
        embedding = embedder.extract(frame, face_info)


    except Exception as e:
        print(f"[VERIFY] Error al extraer embedding: {e}", flush=True)
        ruta_img = capture_manager.save(frame, ci, "error", tipo_marcado)
        db.save_log(persona_id, 0, "error_embedding", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {"match": False, "resultado": "desconocido", "message": "Error al procesar la imagen."}



    embeddings = db.get_embeddings_by_persona(persona_id)


    if not embeddings:
        print("[VERIFY] No hay embeddings registrados para esta persona", flush=True)
        ruta_img = capture_manager.save(frame, ci, "error", tipo_marcado)
        db.save_log(persona_id, 0, "desconocido", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {"match": False, "resultado": "desconocido", "message": "No hay embeddings registrados."}

    result = find_best_match(embedding, embeddings)
    print(f"[VERIFY] Comparación identidad: match={result['match']} confidence={result['confidence']:.2f}%", flush=True)

    if not result["match"]:
        tiempo_ms = int((time.time() - start) * 1000)
        ruta_img = capture_manager.save(frame, ci, "desconocido", tipo_marcado)
        log_id = db.save_log(persona_id, result["confidence"], "desconocido",  # ← CAPTURAR
                tiempo_proceso_ms=tiempo_ms,
                ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {
            "match": False,
            "resultado": "desconocido",
            "confianza": result["confidence"],
            "message": "No se pudo verificar la identidad.",
            "tiempo_proceso_ms": tiempo_ms,
            "log_id": log_id  # ← AGREGAR
        }

    print(f"[VERIFY] ✅ Identidad verificada: {result['confidence']:.2f}%", flush=True)

    # ══════════════════════════════════════════════════════════════
    # PASO 9: Procesar foto_gesto
    # ══════════════════════════════════════════════════════════════
    print("[VERIFY] Paso 4: Procesando foto de gesto...", flush=True)

    if foto_gesto.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Formato de imagen no válido.")

    contents_gesto = foto_gesto.file.read()
    if len(contents_gesto) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Imagen demasiado grande. Máximo 5MB.")

    try:
        pil_gesto = Image.open(BytesIO(contents_gesto))
        pil_gesto.verify()
        pil_gesto = Image.open(BytesIO(contents_gesto))
        if pil_gesto.format not in ("JPEG", "PNG"):
            raise HTTPException(400, "Formato no soportado.")
        frame_gesto = np.ascontiguousarray(np.array(pil_gesto.convert("RGB"))[:, :, ::-1])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "No se pudo procesar la foto de gesto.")

    # ══════════════════════════════════════════════════════════════
    # PASO 10: Detectar rostro en foto_gesto
    # ══════════════════════════════════════════════════════════════
    faces_gesto = detector.detect(frame_gesto)

    if faces_gesto is None:
        print("[VERIFY] No se detectó rostro en foto de gesto", flush=True)
        ruta_img = capture_manager.save(frame_gesto, ci, "error", tipo_marcado)
        db.save_log(persona_id, 0, "desconocido", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {"match": False, "resultado": "desconocido", "message": "No se detectó rostro en foto de gesto."}

    if len(faces_gesto) > 1:
        print(f"[VERIFY] Múltiples rostros en foto de gesto: {len(faces_gesto)}", flush=True)
        ruta_img = capture_manager.save(frame_gesto, ci, "error", tipo_marcado)
        db.save_log(persona_id, 0, "desconocido", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {"match": False, "resultado": "desconocido", "message": "Múltiples rostros en foto de gesto."}
    face_gesto_info = faces_gesto[0]

    # ══════════════════════════════════════════════════════════════
    # PASO 11: Verificar LIVENESS en foto_gesto
    # ══════════════════════════════════════════════════════════════
    print("[VERIFY] Paso 5: Verificando liveness en foto gesto...", flush=True)
    live_result_gesto = liveness.predict(frame_gesto, face_gesto_info)
    print(f"[VERIFY] Resultado liveness gesto: is_real={live_result_gesto['is_real']} score={live_result_gesto['score']:.4f}", flush=True)



    if not live_result_gesto["is_real"]:
        ruta_img = capture_manager.save(frame_gesto, ci, "spoofing_detectado", tipo_marcado)
        log_id = db.save_log(persona_id, 0, "spoofing_detectado",  # ← CAPTURAR
                liveness_score=live_result_gesto["score"],
                ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {
            "match": False,
            "resultado": "spoofing_detectado",
            "message": "Posible suplantación detectada en la foto de gesto.",
            "liveness_score": live_result_gesto["score"],
            "log_id": log_id  # ← AGREGAR
        }

    # ══════════════════════════════════════════════════════════════
    # PASO 12: Verificar LENTES en foto_gesto
    # ══════════════════════════════════════════════════════════════
    print("[VERIFY] Paso 6: Verificando lentes en foto gesto...", flush=True)
    has_eyeglass_gesto = eyeglass_detector.detect(frame_gesto, face_gesto_info)
    print(f"[VERIFY] Resultado lentes gesto: {has_eyeglass_gesto}", flush=True)

    if has_eyeglass_gesto:
        ruta_img = capture_manager.save(frame_gesto, ci, "con_lentes", tipo_marcado)
        db.save_log(persona_id, 0, "con_lentes", ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {
            "match": False,
            "resultado": "desconocido",
            "message": "Por favor, quítese las gafas para la verificación.",
            "eyeglass_detected": True
        }

    # ══════════════════════════════════════════════════════════════
    # PASO 13: Verificar GESTO
    # ══════════════════════════════════════════════════════════════
    print(f"[VERIFY] Paso 7: Verificando gesto solicitado: {gesto_solicitado}...", flush=True)
    gesto_ok = gesture_detector.verify(face_info, face_gesto_info, gesto_solicitado)
    print(f"[VERIFY] Resultado gesto: {gesto_ok}", flush=True)



    if not gesto_ok:
        ruta_img = capture_manager.save(frame_gesto, ci, "gesto_no_coincide", tipo_marcado)
        log_id = db.save_log(persona_id, 0, "gesto_no_coincide",  # ← CAPTURAR
                ip_origen=request.client.host,
                imagen_captura=ruta_img)
        return {
            "match": False,
            "resultado": "gesto_no_coincide",
            "message": f"El gesto '{gesto_solicitado}' no coincide. Por favor, realice el gesto solicitado.",
            "gesture_detected": False,
            "log_id": log_id  # ← AGREGAR
        }



    # ══════════════════════════════════════════════════════════════
    # PASO 14: CORRECTO
    # ══════════════════════════════════════════════════════════════

    tiempo_ms = int((time.time() - start) * 1000)
    resultado = "reconocido"

    ruta_img = capture_manager.save(frame, ci, "reconocido", tipo_marcado)
    log_id = db.save_log(persona_id, result["confidence"], resultado,
                        tiempo_proceso_ms=tiempo_ms,
                        ip_origen=request.client.host,
                        imagen_captura=ruta_img)

    print(f"[VERIFY] ✅ Verificación exitosa: {result['confidence']:.2f}% en {tiempo_ms}ms", flush=True)

    return {
        "match": True,
        "resultado": resultado,
        "confianza": result["confidence"],
        "tiempo_proceso_ms": tiempo_ms,
        "imagen_captura": ruta_img,
        "log_id": log_id
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