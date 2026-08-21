"""
database.py — Conexión PostgreSQL y operaciones CRUD para embeddings.
"""
import psycopg2
import psycopg2.pool
import numpy as np
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    return _pool

def get_connection():
    return get_pool().getconn()

def release_connection(conn):
    get_pool().putconn(conn)


# ── Embeddings ────────────────────────────────────────────────────

def save_embedding(docente_id: int, reconocimiento_facial_id: int, 
                   embedding: np.ndarray, quality_score: float, posicion: str):
    """Guarda un embedding en la tabla embeddings_faciales."""
    conn = get_connection()
    try:
        embedding_bytes = embedding.astype(np.float32).tobytes()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO embeddings_faciales 
                (reconocimiento_facial_id, id_docente, embedding, quality_score, posicion, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (reconocimiento_facial_id, docente_id, embedding_bytes, quality_score, posicion))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def get_embeddings_by_docente(docente_id: int) -> list:
    """Obtiene todos los embeddings de un docente."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, embedding, quality_score, posicion
                FROM embeddings_faciales
                WHERE id_docente = %s
                ORDER BY id
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
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def count_embeddings(docente_id: int) -> int:
    """Cuenta cuántos embeddings tiene un docente."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM embeddings_faciales WHERE id_docente = %s
            """, (docente_id,))
            return cur.fetchone()[0]
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


# ── Reconocimiento Facial ─────────────────────────────────────────

def get_or_create_reconocimiento(docente_id: int) -> int:
    """Obtiene o crea el registro en reconocimiento_facial. Retorna el id."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO reconocimiento_facial (id_docente, activo, total_embeddings, calidad_promedio)
                VALUES (%s, TRUE, 0, 0.000)
                ON CONFLICT (id_docente) DO UPDATE SET id_docente = EXCLUDED.id_docente
                RETURNING id
            """, (docente_id,))
            conn.commit()
            return cur.fetchone()[0]
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def update_reconocimiento_stats(docente_id: int):
    """Actualiza total_embeddings y calidad_promedio."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE reconocimiento_facial SET
                    total_embeddings = (SELECT COUNT(*) FROM embeddings_faciales WHERE id_docente = %s),
                    calidad_promedio = COALESCE((SELECT ROUND(AVG(quality_score)::numeric, 3) FROM embeddings_faciales WHERE id_docente = %s), 0.000),
                    updated_at = NOW()
                WHERE id_docente = %s
            """, (docente_id, docente_id, docente_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def get_reconocimiento_status(docente_id: int) -> dict:
    """Obtiene el estado del registro facial de un docente."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT activo, total_embeddings, calidad_promedio
                FROM reconocimiento_facial WHERE id_docente = %s
            """, (docente_id,))
            row = cur.fetchone()
            if row:
                return {
                    "activo": row[0],
                    "total_embeddings": row[1],
                    "calidad_promedio": float(row[2])
                }
            return {"activo": False, "total_embeddings": 0, "calidad_promedio": 0.0}
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


# ── Log ───────────────────────────────────────────────────────────

def save_log(docente_id: int, confianza: float, resultado: str, 
             liveness_score: float = None, ip_origen: str = None, 
             tiempo_proceso_ms: int = None):
    """Registra un intento de verificación en log_reconocimiento."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO log_reconocimiento 
                (docente_id, confianza, resultado, liveness_score, ip_origen, tiempo_proceso_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (docente_id, confianza, resultado, liveness_score, ip_origen, tiempo_proceso_ms))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)