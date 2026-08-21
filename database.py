"""
database.py — Conexión a dos bases de datos PostgreSQL
- Utic_medicina: para obtener datos de usuarios/docentes
- sistema_asistencia: para almacenar embeddings y logs biométricos
"""
import psycopg2
import psycopg2.pool
import numpy as np
from config import (
    # BD principal (sistema_asistencia)
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    # BD secundaria (Utic_medicina)
    DB2_HOST, DB2_PORT, DB2_NAME, DB2_USER, DB2_PASSWORD
)

# ── Pool de conexiones para sistema_asistencia ──────────────────────
_pool_asistencia = None

def get_pool_asistencia():
    global _pool_asistencia
    if _pool_asistencia is None:
        _pool_asistencia = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    return _pool_asistencia

def get_conn_asistencia():
    return get_pool_asistencia().getconn()

def release_conn_asistencia(conn):
    get_pool_asistencia().putconn(conn)


# ── Conexión directa a Utic_medicina (sin pool, consultas puntuales) ──
def get_conn_utic():
    """Conexión directa a la base de datos Utic_medicina."""
    return psycopg2.connect(
        host=DB2_HOST,
        port=DB2_PORT,
        dbname=DB2_NAME,
        user=DB2_USER,
        password=DB2_PASSWORD
    )


# ── Consultas a Utic_medicina ──────────────────────────────────────

def get_docente_by_ci(ci: str) -> dict:
    """
    Obtiene los datos del docente desde Utic_medicina.
    Busca en users y docente.
    Retorna: {id, ci, email, nombre?, departamento, estado}
    """
    conn = get_conn_utic()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    u.id AS user_id,
                    u.ci,
                    u.email,
                    u.rol,
                    d.id AS docente_id,
                    d.departamento,
                    d.estado
                FROM public.users u
                INNER JOIN public.docente d ON d.id_user = u.id
                WHERE u.ci = %s
            """, (ci,))
            row = cur.fetchone()
            
            if not row:
                return None
            
            return {
                "user_id": row[0],
                "ci": row[1],
                "email": row[2],
                "rol": row[3],
                "docente_id": row[4],  # Este es el ID que usaremos en sistema_asistencia
                "departamento": row[5],
                "estado": row[6]
            }
    finally:
        conn.close()


def get_docente_by_id(docente_id: int) -> dict:
    """Obtiene datos del docente por su ID en Utic_medicina."""
    conn = get_conn_utic()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    u.id AS user_id,
                    u.ci,
                    u.email,
                    u.rol,
                    d.id AS docente_id,
                    d.departamento,
                    d.estado
                FROM public.users u
                INNER JOIN public.docente d ON d.id_user = u.id
                WHERE d.id = %s
            """, (docente_id,))
            row = cur.fetchone()
            
            if not row:
                return None
            
            return {
                "user_id": row[0],
                "ci": row[1],
                "email": row[2],
                "rol": row[3],
                "docente_id": row[4],
                "departamento": row[5],
                "estado": row[6]
            }
    finally:
        conn.close()


def verificar_docente_existe(docente_id: int) -> bool:
    """Verifica si un docente existe en Utic_medicina."""
    conn = get_conn_utic()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM public.docente WHERE id = %s", (docente_id,))
            return cur.fetchone() is not None
    finally:
        conn.close()


# ── Operaciones en sistema_asistencia ──────────────────────────────

def get_or_create_persona(docente_id: int, ci: str) -> int:
    """
    Obtiene o crea una persona en sistema_asistencia.
    Retorna el id de la tabla persona.
    """
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            # Verificar si existe
            cur.execute("""
                SELECT id FROM public.persona WHERE docente_id = %s
            """, (docente_id,))
            row = cur.fetchone()
            
            if row:
                return row[0]
            
            # Crear nueva persona
            cur.execute("""
                INSERT INTO public.persona (ci, docente_id, fecha_registro, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW(), NOW())
                RETURNING id
            """, (ci, docente_id))
            persona_id = cur.fetchone()[0]
            conn.commit()
            return persona_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn_asistencia(conn)


def get_persona_by_docente_id(docente_id: int) -> dict:
    """Obtiene los datos de persona en sistema_asistencia por docente_id."""
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ci, docente_id, fecha_registro, created_at, updated_at
                FROM public.persona
                WHERE docente_id = %s
            """, (docente_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "ci": row[1],
                "docente_id": row[2],
                "fecha_registro": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            }
    finally:
        release_conn_asistencia(conn)


# ── Embeddings (sistema_asistencia) ──────────────────────────────────

def save_embedding(docente_id: int, embedding: np.ndarray, 
                   quality_score: float, posicion: str) -> int:
    """
    Guarda un embedding en sistema_asistencia.
    Primero obtiene/crea la persona y el reconocimiento_facial.
    """
    # 1. Obtener el CI del docente desde Utic_medicina
    docente = get_docente_by_id(docente_id)
    if not docente:
        raise ValueError(f"Docente con ID {docente_id} no encontrado en Utic_medicina")
    
    ci = docente["ci"]
    
    # 2. Obtener o crear persona en sistema_asistencia
    persona_id = get_or_create_persona(docente_id, ci)
    
    # 3. Obtener o crear reconocimiento_facial
    recon_id = get_or_create_reconocimiento(persona_id)
    
    # 4. Guardar embedding
    conn = get_conn_asistencia()
    try:
        embedding_bytes = embedding.astype(np.float32).tobytes()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.embeddings_faciales 
                (reconocimiento_facial_id, persona_id, embedding, quality_score, posicion, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING id
            """, (recon_id, persona_id, embedding_bytes, quality_score, posicion))
            embedding_id = cur.fetchone()[0]
        conn.commit()
        
        # 5. Actualizar estadísticas
        update_reconocimiento_stats(persona_id)
        
        return embedding_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn_asistencia(conn)


def get_embeddings_by_docente(docente_id: int) -> list:
    """Obtiene todos los embeddings de un docente desde sistema_asistencia."""
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.embedding, e.quality_score, e.posicion
                FROM public.embeddings_faciales e
                INNER JOIN public.persona p ON p.id = e.persona_id
                WHERE p.docente_id = %s
                ORDER BY e.id
            """, (docente_id,))
            rows = cur.fetchall()
        
        result = []
        for row in rows:
            emb = np.frombuffer(row[1], dtype=np.float32)
            result.append({
                "id": row[0],
                "embedding": emb,
                "quality_score": float(row[2]),
                "posicion": row[3]
            })
        return result
    finally:
        release_conn_asistencia(conn)


def count_embeddings(docente_id: int) -> int:
    """Cuenta cuántos embeddings tiene un docente."""
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM public.embeddings_faciales e
                INNER JOIN public.persona p ON p.id = e.persona_id
                WHERE p.docente_id = %s
            """, (docente_id,))
            return cur.fetchone()[0]
    finally:
        release_conn_asistencia(conn)


# ── Reconocimiento Facial (sistema_asistencia) ──────────────────────

def get_or_create_reconocimiento(persona_id: int) -> int:
    """Obtiene o crea el registro en reconocimiento_facial."""
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            # Verificar si existe
            cur.execute("""
                SELECT id FROM public.reconocimiento_facial WHERE persona_id = %s
            """, (persona_id,))
            row = cur.fetchone()
            
            if row:
                return row[0]
            
            # Crear nuevo
            cur.execute("""
                INSERT INTO public.reconocimiento_facial 
                (persona_id, activo, total_embeddings, calidad_promedio, created_at, updated_at)
                VALUES (%s, TRUE, 0, 0.000, NOW(), NOW())
                RETURNING id
            """, (persona_id,))
            recon_id = cur.fetchone()[0]
            conn.commit()
            return recon_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn_asistencia(conn)


def update_reconocimiento_stats(persona_id: int):
    """Actualiza total_embeddings y calidad_promedio en reconocimiento_facial."""
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE public.reconocimiento_facial SET
                    total_embeddings = (
                        SELECT COUNT(*) FROM public.embeddings_faciales 
                        WHERE persona_id = %s
                    ),
                    calidad_promedio = COALESCE((
                        SELECT ROUND(AVG(quality_score)::numeric, 3) 
                        FROM public.embeddings_faciales 
                        WHERE persona_id = %s
                    ), 0.000),
                    updated_at = NOW()
                WHERE persona_id = %s
            """, (persona_id, persona_id, persona_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn_asistencia(conn)


def get_reconocimiento_status(docente_id: int) -> dict:
    """Obtiene el estado del reconocimiento facial de un docente."""
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.activo, r.total_embeddings, r.calidad_promedio
                FROM public.reconocimiento_facial r
                INNER JOIN public.persona p ON p.id = r.persona_id
                WHERE p.docente_id = %s
            """, (docente_id,))
            row = cur.fetchone()
            if row:
                return {
                    "activo": row[0],
                    "total_embeddings": row[1],
                    "calidad_promedio": float(row[2])
                }
            return {"activo": False, "total_embeddings": 0, "calidad_promedio": 0.0}
    finally:
        release_conn_asistencia(conn)


# ── Log (sistema_asistencia) ─────────────────────────────────────────

def save_log(docente_id: int, confianza: float, resultado: str, 
             liveness_score: float = None, ip_origen: str = None, 
             tiempo_proceso_ms: int = None, imagen_captura: str = None,
             dispositivo_id: str = None):
    """
    Registra un intento de verificación en log_reconocimiento.
    """
    conn = get_conn_asistencia()
    try:
        # Buscar persona_id
        persona = get_persona_by_docente_id(docente_id)
        persona_id = persona["id"] if persona else None
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.log_reconocimiento 
                (persona_id, confianza, resultado, liveness_score, ip_origen, 
                 dispositivo_id, imagen_captura, tiempo_proceso_ms, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (persona_id, confianza, resultado, liveness_score, ip_origen, 
                  dispositivo_id, imagen_captura, tiempo_proceso_ms))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn_asistencia(conn)


# ── Función de utilidad ──────────────────────────────────────────────

def get_docente_info(ci: str = None, docente_id: int = None) -> dict:
    """
    Función unificada para obtener información de un docente.
    Busca por CI o por ID.
    """
    if ci:
        return get_docente_by_ci(ci)
    elif docente_id:
        return get_docente_by_id(docente_id)
    return None


def persona_existe_en_asistencia(docente_id: int) -> bool:
    """Verifica si un docente ya tiene registro en sistema_asistencia."""
    conn = get_conn_asistencia()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM public.persona WHERE docente_id = %s", (docente_id,))
            return cur.fetchone() is not None
    finally:
        release_conn_asistencia(conn)