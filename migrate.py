"""
migrate.py — Crea las tablas en Servicio_reconocimiento.
Uso: python migrate.py
"""
import psycopg2
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

SQL = """
CREATE TABLE IF NOT EXISTS persona (
    id              BIGSERIAL       PRIMARY KEY,
    ci              VARCHAR(20)     NOT NULL UNIQUE,
    fecha_registro  TIMESTAMP(0)    DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_at      TIMESTAMP(0),
    updated_at      TIMESTAMP(0)
);

CREATE INDEX IF NOT EXISTS persona_fecha_registro_index ON persona(fecha_registro);

CREATE TABLE IF NOT EXISTS reconocimiento_facial (
    id                  BIGSERIAL       PRIMARY KEY,
    persona_id          BIGINT          NOT NULL UNIQUE,
    activo              BOOLEAN         DEFAULT TRUE NOT NULL,
    total_embeddings    SMALLINT        DEFAULT 0 NOT NULL,
    calidad_promedio    NUMERIC(4,3)    DEFAULT 0.000 NOT NULL,
    created_at          TIMESTAMP(0),
    updated_at          TIMESTAMP(0),
    CONSTRAINT fk_reconocimiento_persona
        FOREIGN KEY (persona_id) REFERENCES persona(id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS embeddings_faciales (
    id                          BIGSERIAL       PRIMARY KEY,
    reconocimiento_facial_id    BIGINT          NOT NULL,
    persona_id                  BIGINT          NOT NULL,
    embedding                   BYTEA           NOT NULL,
    quality_score               NUMERIC(4,3)    DEFAULT 0.000 NOT NULL,
    posicion                    VARCHAR(255)    NOT NULL,
    created_at                  TIMESTAMP(0),
    updated_at                  TIMESTAMP(0),
    CONSTRAINT chk_posicion CHECK (posicion IN ('centro','izquierda','derecha','arriba','abajo','sonrisa')),
    CONSTRAINT fk_embedding_reconocimiento
        FOREIGN KEY (reconocimiento_facial_id) REFERENCES reconocimiento_facial(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_embedding_persona
        FOREIGN KEY (persona_id) REFERENCES persona(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_embeddings_persona ON embeddings_faciales(persona_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_reconocimiento ON embeddings_faciales(reconocimiento_facial_id);

CREATE TABLE IF NOT EXISTS log_reconocimiento (
    id                  BIGSERIAL       PRIMARY KEY,
    persona_id          BIGINT          NULL,
    confianza           NUMERIC(5,2)    DEFAULT 0.00 NOT NULL,
    resultado           VARCHAR(255)    NOT NULL,
    liveness_score      NUMERIC(4,3),
    ip_origen           VARCHAR(45),
    dispositivo_id      VARCHAR(255),
    imagen_captura      VARCHAR(500),
    tiempo_proceso_ms   SMALLINT,
    created_at          TIMESTAMP(0),
    updated_at          TIMESTAMP(0),
    CONSTRAINT chk_resultado CHECK (resultado IN ('reconocido','desconocido','spoofing_detectado','gesto_no_coincide','error_embedding')),
    CONSTRAINT fk_log_persona
        FOREIGN KEY (persona_id) REFERENCES persona(id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_log_persona ON log_reconocimiento(persona_id);
CREATE INDEX IF NOT EXISTS idx_log_created_at ON log_reconocimiento(created_at);
"""

def migrate():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(SQL)
    conn.close()
    print("✅ Migraciones ejecutadas correctamente.")

if __name__ == "__main__":
    migrate()