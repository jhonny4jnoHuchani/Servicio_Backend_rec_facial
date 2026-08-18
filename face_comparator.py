"""
face_comparator.py — Comparación de embeddings por docente.
"""
import numpy as np
from config import COSINE_THRESHOLD

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Similitud coseno entre dos embeddings. Rango [0, 1]."""
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return float(dot / norm) if norm > 0 else 0.0

def find_best_match(query: np.ndarray, embeddings: list) -> dict:
    """
    Compara un embedding contra una lista de embeddings del docente.
    Retorna el mejor match con su score.
    """
    if not embeddings:
        return {"match": False, "confidence": 0.0, "embedding_id": None}

    best_score = -1.0
    best_id = None

    for entry in embeddings:
        score = cosine_similarity(query, entry["embedding"])
        if score > best_score:
            best_score = score
            best_id = entry["id"]

    confidence = round(best_score * 100.0, 2)
    match = best_score >= COSINE_THRESHOLD

    return {
        "match": match,
        "confidence": confidence,
        "embedding_id": best_id
    }