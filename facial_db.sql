-- ============================================================
-- BASE DE DATOS: Servicio_reconocimiento
-- Backup SQL para servidores
-- ============================================================

CREATE DATABASE "Servicio_reconocimiento";
\c "Servicio_reconocimiento";

CREATE TABLE persona (
    id              BIGSERIAL       PRIMARY KEY,
    ci              VARCHAR(20)     NOT NULL UNIQUE,
    fecha_registro  TIMESTAMP(0)    DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_at      TIMESTAMP(0),
    updated_at      TIMESTAMP(0)
);

CREATE INDEX persona_fecha_registro_index ON persona(fecha_registro);

CREATE TABLE reconocimiento_facial (
    id                  BIGSERIAL       PRIMARY KEY,
    persona_id          BIGINT          NOT NULL UNIQUE,
    activo              BOOLEAN         DEFAULT TRUE NOT NULL,
    total_embeddings    SMALLINT        DEFAULT 0 NOT NULL,
    calidad_promedio    NUMERIC(4,3)    DEFAULT 0.000 NOT NULL,
    created_at          TIMESTAMP(0),
    updated_at          TIMESTAMP(0),
    CONSTRAINT fk_reconocimiento_persona FOREIGN KEY (persona_id) REFERENCES persona(id) ON DELETE RESTRICT
);

CREATE TABLE embeddings_faciales (
    id                          BIGSERIAL       PRIMARY KEY,
    reconocimiento_facial_id    BIGINT          NOT NULL,
    persona_id                  BIGINT          NOT NULL,
    embedding                   BYTEA           NOT NULL,
    quality_score               NUMERIC(4,3)    DEFAULT 0.000 NOT NULL,
    posicion                    VARCHAR(255)    NOT NULL,
    created_at                  TIMESTAMP(0),
    updated_at                  TIMESTAMP(0),
    CONSTRAINT chk_posicion CHECK (posicion IN ('centro','izquierda','derecha','arriba','abajo','sonrisa')),
    CONSTRAINT fk_embedding_reconocimiento FOREIGN KEY (reconocimiento_facial_id) REFERENCES reconocimiento_facial(id) ON DELETE CASCADE,
    CONSTRAINT fk_embedding_persona FOREIGN KEY (persona_id) REFERENCES persona(id) ON DELETE RESTRICT
);

CREATE INDEX idx_embeddings_persona ON embeddings_faciales(persona_id);
CREATE INDEX idx_embeddings_reconocimiento ON embeddings_faciales(reconocimiento_facial_id);

CREATE TABLE log_reconocimiento (
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
    CONSTRAINT chk_resultado CHECK (resultado IN ('reconocido','desconocido','spoofing_detectado')),
    CONSTRAINT fk_log_persona FOREIGN KEY (persona_id) REFERENCES persona(id) ON DELETE SET NULL
);

CREATE INDEX idx_log_persona ON log_reconocimiento(persona_id);
CREATE INDEX idx_log_created_at ON log_reconocimiento(created_at);