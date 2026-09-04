# Servicio de Reconocimiento Facial

Servicio backend desarrollado en **Python (FastAPI)** para verificación biométrica facial, diseñado para un sistema de asistencia docente. Incluye anti-spoofing (liveness detection), detección de lentes y verificación de gestos aleatorios.

---

## 📋 Descripción

Este servicio permite registrar y verificar la identidad de personas mediante reconocimiento facial, incorporando múltiples capas de seguridad para evitar suplantaciones (fotos, pantallas, videos) y garantizar que la verificación se realiza en tiempo real sobre una persona presente físicamente.

---

## 🚀 Tecnologías

| Componente | Versión |
|------------|---------|
| Python | 3.12+ |
| FastAPI | 0.115+ |
| OpenCV | 4.10+ |
| PostgreSQL | 16+ |
| psycopg2 | 2.9+ |
| Pillow | 10+ |
| Uvicorn | 0.30+ |

---

## 📁 Estructura del Proyecto

```
facial-service/
├── middleware/
│   ├── __init__.py
│   └── auth.py                  # Validación API Key
├── models/                      # Modelos ONNX
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── 2.7_80x80_MiniFASNetV2.onnx
├── capturas/                    # Imágenes de verificación (WebP)
├── logs/                        # Logs por día
├── sospechosas/                 # Imágenes de spoofing detectado
├── temp/                        # Archivos temporales
├── capture_manager.py           # Gestor de capturas
├── config.py                    # Configuración general
├── database.py                  # Conexión PostgreSQL
├── face_comparator.py           # Comparación de embeddings
├── face_detector.py             # Detección facial (YuNet)
├── face_embedder.py             # Extracción de embeddings (SFace)
├── eyeglass_detector.py         # Detección de lentes
├── gesture_detector.py          # Detección de gestos
├── liveness_detector.py         # Anti-spoofing (MiniFASNet)
├── logger.py                    # Logger de verificaciones
├── main.py                      # Endpoints FastAPI
├── migrate.py                   # Creación de tablas
├── requirements.txt             # Dependencias
└── .env                         # Variables de entorno
```

---

## 🔌 Endpoints

| Método | Ruta | Descripción | Auth |
|--------|------|--------------|------|
| POST | `/register-persona` | Crear/buscar persona por CI | ✅ API Key |
| POST | `/register` | Registrar embedding facial | ✅ API Key |
| POST | `/verify` | Verificar rostro + gesto | ✅ API Key |
| GET | `/status/{persona_id}` | Consultar estado facial de una persona | ✅ API Key |
| GET | `/health` | Health check del servicio | ❌ Público |

---

## 🔑 Seguridad

- **API Key** obligatoria en el header `X-API-Key` para todos los endpoints protegidos.
- **IP Whitelist** opcional (deshabilitada por defecto, se puede activar en `.env`).

```env
API_KEY=clave_secreta_super_segura_2026
# ALLOWED_IPS=127.0.0.1
```

---

## 🗄️ Base de Datos

**Nombre:** `Servicio_reconocimiento`

| Tabla | Descripción |
|-------|-------------|
| `persona` | Personas registradas por CI (`id`, `ci`, `fecha_registro`) |
| `reconocimiento_facial` | Estado facial de la persona (`persona_id`, `activo`, `total_embeddings`, `calidad_promedio`) |
| `embeddings_faciales` | Vectores faciales 128D (`persona_id`, `embedding` BYTEA, `quality_score`, `posicion`) |
| `log_reconocimiento` | Historial de verificaciones (`persona_id`, `confianza`, `resultado`, `liveness_score`, `imagen_captura`) |

---

## 📸 Capturas

- **Formato:** WebP (calidad 80)
- **Estructura de carpetas:**
  ```
  capturas/{CI}/{fecha_dd-mm-aaaa_Día}/{tipo}_{resultado}_{hora}.webp
  ```

---

## 📝 Logs

- **Formato:** 2 líneas por verificación
- **Archivo:** `logs/verify-YYYY-MM-DD.log`

**Ejemplo:**
```
[VERIFY] persona_id=11 | ci=9899401 | gesto=izquierda | tipo=entrada
[RESULTADO] reconocido | log_id=220 | imagen=... | 128ms
```

---

## ⚙️ Configuración (.env)

```env
# Servidor
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8001

# Base de datos
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=Servicio_reconocimiento
DB_USER=postgres
DB_PASSWORD=

# Detección de lentes
EYEGLASS_DETECTION_ENABLED=true
EYEGLASS_THRESHOLD=0.7
EYEGLASS_MODEL_SIZE=small
EYEGLASS_KIND=anyglasses

# Capturas
SAVE_CAPTURES_ENABLED=true
SAVE_RECONOCIDO=true

# Seguridad
API_KEY=clave_secreta_super_segura_2026
# ALLOWED_IPS=127.0.0.1
```

---

## 🚀 Instalación

```bash
# 1. Clonar repositorio
git clone <url>

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual (Windows)
venv\Scripts\activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Copiar archivo de entorno
copy .env.example .env

# 6. Configurar el archivo .env con tus datos

# 7. Ejecutar migraciones
python migrate.py

# 8. Iniciar el servicio
python main.py
```

---



---

## 📊 Resultados de Verificación

| Resultado | Descripción |
|-----------|-------------|
| `reconocido` | Persona verificada correctamente |
| `desconocido` | No coincide con embeddings registrados |
| `spoofing_detectado` | Foto o pantalla detectada (anti-liveness) |
| `gesto_no_coincide` | El gesto realizado no coincide con el solicitado |
| `lentes` | Usuario detectado usando gafas |

---

## 🔒 Capas de Seguridad Anti-Spoofing

| Capa | Tecnología |
|------|------------|
| Liveness | MiniFASNet |
| Lentes | Glasses-detector |
| Gesto | YuNet landmarks |
| API Key | Header `X-API-Key` |

---

## 📄 Licencia

Proyecto interno — uso restringido al sistema de asistencia docente.