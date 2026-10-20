# Servicio Backend de Reconocimiento Facial y Verificación Biométrica

Microservicio backend de alto rendimiento desarrollado en **Python (FastAPI)** para la verificación biométrica e identificación facial en tiempo real, diseñado para el sistema de control de asistencia docente.

El servicio implementa una arquitectura multicapa de seguridad que combina **reconocimiento facial con Deep Learning (SFace)**, **detección de rostros y landmarks (YuNet)**, **anti-spoofing pasivo (MiniFASNetV2)**, **anti-spoofing activo por gestos angulares/faciales**, y **detección de lentes (Glasses-Detector)**.

---

## 📑 Tabla de Contenidos

- [Características Principales](#-características-principales)
- [Arquitectura de Seguridad e Inteligencia Artificial](#-arquitectura-de-seguridad-e-inteligencia-artificial)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos y Dependencias](#-requisitos-y-dependencias)
- [Base de Datos (PostgreSQL)](#-base-de-datos-postgresql)
- [Variables de Entorno (.env)](#-variables-de-entorno-env)
- [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
- [Pipeline de Verificación Paso a Paso](#-pipeline-de-verificación-paso-a-paso)
- [Catálogo de Endpoints de la API](#-catálogo-de-endpoints-de-la-api)
- [Herramienta de Registro Facial (GUI)](#-herramienta-de-registro-facial-gui)
- [Almacenamiento de Capturas y Evidencias](#-almacenamiento-de-capturas-y-evidencias)
- [Sistema de Logs y Auditoría](#-sistema-de-logs-y-auditoría)
- [Buenas Prácticas y Producción](#-buenas-prácticas-y-producción)

---

## 🌟 Características Principales

- **Identificación 1:N / Verificación 1:1 robusta:** Comparación vectorial mediante distancia coseno contra embeddings pre-calculados de cada persona.
- **Detección facial de alta velocidad:** Ejecución de YuNet optimizado con OpenCV DNN sobre modelos ONNX sin requerir frameworks pesados en inferencia de rostros.
- **Anti-Spoofing Pasivo (Liveness Detection):** Detección de fotos impresas, pantallas de smartphones, tablets o monitores usando MiniFASNetV2.
- **Anti-Spoofing Activo (Challenge-Response por Gestos):** Validación de cambios angulares 3D de la cabeza (*yaw* y *pitch*) y ensanchamiento de comisuras de la boca (*sonrisa*) entre la captura frontal y la foto de desafío.
- **Detección de Lentes Obligatoria:** Rechazo automático preventivo si el usuario intenta registrarse o verificarse portando lentes/gafas, garantizando la consistencia ocular del embedding.
- **Almacenamiento Eficiente en WebP:** Compresión automática de imágenes de evidencia a formato WebP (calidad 80), organizadas por CI y fecha con nombres legibles.
- **Protección por API Key e IP:** Middleware de autenticación centralizado para proteger todos los endpoints transaccionales.

---

## 🧠 Arquitectura de Seguridad e Inteligencia Artificial

El sistema combina cuatro tecnologías especializadas:

| Capa | Modelo / Herramienta | Archivo / Entrada | Función Principal | Umbral por Defecto |
| :--- | :--- | :--- | :--- | :--- |
| **Detección Facial** | **YuNet** (ONNX) | `models/face_detection_yunet_2023mar.onnx` | Detección rápida de rostro, bounding box y 5 landmarks (ojos, nariz, comisuras boca). | Score $\ge 0.60$, tamaño mín. $60\times 60$ px |
| **Embeddings Faciales** | **SFace** (ONNX) | `models/face_recognition_sface_2021dec.onnx` | Alineación facial (`alignCrop`) y extracción de vector normalizado de 128 dimensiones (float32). | Similitud coseno $\ge 0.40$ |
| **Anti-Spoofing Pasivo** | **MiniFASNetV2** (ONNX) | `models/2.7_80x80_MiniFASNetV2.onnx` | Clasificación de liveness con recorte $2.7\times$ a $80\times 80$ px y activación Softmax sobre la clase real. | Probabilidad real $\ge 0.50$ |
| **Detección de Lentes** | **GlassesClassifier** | PyTorch / Torchvision (`glasses-detector`) | Clasificador binario que identifica si la persona porta gafas graduadas o de sol. Cuenta con caché TTL y timeout protector. | Probabilidad $> 0.70$ |
| **Gesto Biométrico** | Algoritmos Biométricos | Landmark ratios en `gesture_detector.py` | Cálculo cinemático de rotaciones (`arriba`, `abajo`, `izquierda`, `derecha`, `sonrisa`) comparando ambas capturas. | Calibrado con deltas de pitch, yaw y ancho de boca |

---

## 📁 Estructura del Proyecto

```text
facial-service/
├── capturas/                     # Directorio de evidencias fotográficas en formato WebP
├── logs/                         # Archivos de registro diario (verify-YYYY-MM-DD.log)
├── middleware/
│   ├── __init__.py
│   └── auth.py                   # Middleware de autenticación (Header X-API-Key / IPs)
├── models/                       # Modelos neuronales en formato ONNX
│   ├── 2.7_80x80_MiniFASNetV2.onnx
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── sospechosas/                  # Capturas de suplantación/spoofing detectadas en registro
├── temp/                         # Directorio para operaciones y archivos temporales
├── .env                          # Variables de entorno activas (no subir a git)
├── .env.example                  # Plantilla de variables de entorno
├── capture_manager.py            # Gestor de compresión y jerarquía de almacenamiento WebP
├── config.py                     # Configuración central, umbrales y carga de variables
├── database.py                   # Pool de conexiones ThreadedConnectionPool y consultas PostgreSQL
├── eyeglass_detector.py          # Wrapper de detección de lentes con caché y timeout
├── face_comparator.py            # Comparación vectorial coseno entre embeddings
├── face_detector.py              # Wrapper de OpenCV YuNet para inferencia facial
├── face_embedder.py              # Wrapper de OpenCV SFace para extracción de vectores 128D
├── facial_db.sql                 # Script SQL de creación de la base de datos y esquemas
├── gesture_detector.py           # Detección matemática de gestos por landmarks faciales
├── liveness_detector.py          # Inferencia de anti-spoofing con MiniFASNetV2
├── logger.py                     # Configuración del logger de auditoría
├── main.py                       # Aplicación FastAPI y definición de endpoints REST
├── migrate.py                    # Script de migración y creación de tablas en PostgreSQL
├── register_ui.py                # Aplicación GUI de escritorio (Tkinter) para enrolamiento
├── requirements.txt              # Lista de dependencias del entorno Python
└── README.md                     # Documentación general del sistema
```

---

## 📦 Requisitos y Dependencias

- **Sistema Operativo:** Windows 10/11, Linux (Ubuntu 20.04+) o macOS.
- **Python:** 3.12 o superior.
- **Base de Datos:** PostgreSQL 16 o superior.
- **Cámara Web:** Resolución mínima recomendada 640x480 a 30 FPS.

### Dependencias Principales (`requirements.txt`)
- `fastapi==0.140.0`: Framework web asíncrono para los endpoints REST.
- `uvicorn==0.51.0`: Servidor ASGI de alto rendimiento.
- `opencv-python-headless==5.0.0.93`: Procesamiento de imágenes y OpenCV DNN.
- `psycopg2-binary==2.9.12`: Driver PostgreSQL con pooling de conexiones.
- `torch==2.13.0` & `torchvision==0.28.0`: Motor de inferencia para detección de lentes.
- `glasses-detector==1.0.4`: Red neuronal de clasificación de lentes.
- `pillow==12.3.0`: Manipulación y validación de buffers de imagen.
- `python-dotenv==1.2.2`: Carga de variables de configuración desde `.env`.
- `requests==2.34.2`: Cliente HTTP utilizado por `register_ui.py`.

---

## 🗄️ Base de Datos (PostgreSQL)

La base de datos utilizada por el servicio se denomina **`Servicio_reconocimiento`**.

```
 ┌──────────────┐       1:1       ┌────────────────────────┐
 │   persona    │─────────────────│ reconocimiento_facial  │
 └──────┬───────┘                 └───────────┬────────────┘
        │                                     │ 1:N
        │ 1:N                                 │
        ├─────────────────────────────────────┘
        │                                 │
        ▼ 1:N                             ▼
 ┌──────────────────────┐        ┌─────────────────────┐
 │  log_reconocimiento  │        │ embeddings_faciales │
 └──────────────────────┘        └─────────────────────┘
```

### 1. Tabla `persona`
Registra la identidad base de los usuarios enrolados.
- `id` (BIGSERIAL, PK): Identificador único de la persona.
- `ci` (VARCHAR(20), UNIQUE, NOT NULL): Cédula de identidad (ej: `"9899401"`).
- `fecha_registro` (TIMESTAMP(0), DEFAULT NOW()): Fecha del registro inicial.
- `created_at`, `updated_at`: Marcas de tiempo de auditoría.

### 2. Tabla `reconocimiento_facial`
Estado biométrico consolidado de la persona.
- `id` (BIGSERIAL, PK): Identificador del registro.
- `persona_id` (BIGINT, UNIQUE, FK -> `persona.id`): Relación 1:1 con la persona.
- `activo` (BOOLEAN, DEFAULT TRUE): Indica si el usuario está habilitado para verificar.
- `total_embeddings` (SMALLINT, DEFAULT 0): Cantidad de vectores registrados.
- `calidad_promedio` (NUMERIC(4,3), DEFAULT 0.000): Calidad promedio de las capturas enroladas.
- `created_at`, `updated_at`: Marcas de tiempo.

### 3. Tabla `embeddings_faciales`
Almacena los vectores normalizados de 128 dimensiones extraídos durante el enrolamiento.
- `id` (BIGSERIAL, PK): Identificador del embedding.
- `reconocimiento_facial_id` (BIGINT, FK -> `reconocimiento_facial.id` ON DELETE CASCADE).
- `persona_id` (BIGINT, FK -> `persona.id` ON DELETE RESTRICT).
- `embedding` (BYTEA, NOT NULL): Vector binario de $128 \times 4$ bytes (float32).
- `quality_score` (NUMERIC(4,3)): Score de confianza asignado por YuNet al detectar el rostro.
- `posicion` (VARCHAR(255), NOT NULL): Pose del rostro (`centro`, `izquierda`, `derecha`, `arriba`, `abajo`, `sonrisa`).
- `created_at`, `updated_at`: Marcas de tiempo.

### 4. Tabla `log_reconocimiento`
Auditoría histórica de todos los intentos de verificación.
- `id` (BIGSERIAL, PK): Identificador del log.
- `persona_id` (BIGINT, NULL, FK -> `persona.id` ON DELETE SET NULL): Persona evaluada.
- `confianza` (NUMERIC(5,2), NOT NULL): Porcentaje de similitud coseno obtenido (0.00% a 100.00%).
- `resultado` (VARCHAR(255), NOT NULL): Estado final (`reconocido`, `desconocido`, `spoofing_detectado`, `gesto_no_coincide`, `error_embedding`).
- `liveness_score` (NUMERIC(4,3)): Score de anti-spoofing pasivo obtenido.
- `ip_origen` (VARCHAR(45)): Dirección IP del cliente que realizó la petición.
- `dispositivo_id` (VARCHAR(255)): Identificador opcional del dispositivo.
- `imagen_captura` (VARCHAR(500)): Ruta relativa del archivo WebP guardado como evidencia.
- `tiempo_proceso_ms` (SMALLINT): Duración total de la verificación en milisegundos.
- `created_at`, `updated_at`: Marcas de tiempo.

---

## ⚙️ Variables de Entorno (.env)

Copia el archivo `.env.example` a `.env` y ajusta según tu entorno:

```env
# ── Servidor ──────────────────────────────────────────────────────────
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8001

# ── Base de datos PostgreSQL ──────────────────────────────────────────
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=Servicio_reconocimiento
DB_USER=postgres
DB_PASSWORD=tu_password_aqui

# ── Detección de Lentes ───────────────────────────────────────────────
EYEGLASS_DETECTION_ENABLED=true
EYEGLASS_THRESHOLD=0.7              # 0.0 - 1.0 (mayor = más estricto)
EYEGLASS_MODEL_SIZE=small           # small, medium, large
EYEGLASS_KIND=anyglasses            # anyglasses, sunglasses, eyeglasses
EYEGLASS_CACHE_SIZE=100             # Cantidad máxima de hashes en caché
EYEGLASS_CACHE_TTL=5                # Segundos de vida de la entrada en caché
EYEGLASS_PREDICTION_TIMEOUT=1.0     # Límite de tiempo en segundos

# ── Gestión de Capturas y Evidencias ──────────────────────────────────
SAVE_CAPTURES_ENABLED=true          # Guardar imágenes de verificación en disco
SAVE_RECONOCIDO=true                # Guardar capturas de verificaciones exitosas

# ── Seguridad y Autenticación ─────────────────────────────────────────
API_KEY=clave_secreta_super_segura_2026
# ALLOWED_IPS=127.0.0.1             # Separadas por comas (opcional)
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio y acceder a la carpeta
```bash
git clone <URL_DEL_REPOSITORIO>
cd facial-service
```

### 2. Crear y activar el entorno virtual
En Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```
En Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar presencia de los modelos ONNX
Asegúrate de que la carpeta `models/` contenga los 3 archivos ONNX:
- `models/face_detection_yunet_2023mar.onnx`
- `models/face_recognition_sface_2021dec.onnx`
- `models/2.7_80x80_MiniFASNetV2.onnx`

### 5. Configurar el archivo `.env`
```bash
copy .env.example .env     # En Windows
# cp .env.example .env     # En Linux
```
Edita `.env` con las credenciales correspondientes a tu servidor PostgreSQL y tu `API_KEY`.

### 6. Ejecutar migraciones de Base de Datos
Asegúrate de haber creado previamente la base de datos `Servicio_reconocimiento` en PostgreSQL y luego ejecuta:
```bash
python migrate.py
```
*Salida esperada:* `✅ Migraciones ejecutadas correctamente.`

### 7. Iniciar el servicio backend
```bash
python main.py
```
O directamente con `uvicorn`:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

El servicio estará disponible en `http://localhost:8001`. Puedes consultar la documentación interactiva Swagger en:
- `http://localhost:8001/docs`
- `http://localhost:8001/redoc`

---

## 🔄 Pipeline de Verificación Paso a Paso

El endpoint principal `POST /verify` ejecuta un flujo estricto de **14 pasos** para garantizar la autenticidad del usuario:

```
[Cliente]
   │
   ├── (1) Validar Persona y CI en Base de Datos
   ├── (2) Validar Mínimo de 70 Embeddings Registrados
   ├── (3) Validar Gesto Solicitado (arriba, abajo, izquierda, derecha, sonrisa)
   │
   ├── [FOTO FRONTAL]
   │     ├── (4) Decodificar y Validar Formato/Tamaño (JPG/PNG <= 5MB)
   │     ├── (5) Detectar Rostro Único con YuNet (si 0 o >1 -> Error)
   │     ├── (6) Detección de Lentes (si usa lentes -> Rechazar con mensaje)
   │     ├── (7) Anti-Spoofing MiniFASNetV2 (score >= 0.50 -> si no, Spoofing)
   │     └── (8) Extraer Embedding 128D y Comparar Coseno con BD (umbral >= 0.40)
   │
   ├── [FOTO DE GESTO]
   │     ├── (9) Decodificar y Validar Formato/Tamaño
   │     ├── (10) Detectar Rostro Único con YuNet
   │     ├── (11) Anti-Spoofing MiniFASNetV2 en Foto Gesto
   │     ├── (12) Detección de Lentes en Foto Gesto
   │     └── (13) Verificar Gesto Biométrico comparando Landmarks Frontal vs Gesto
   │
   └── (14) ÉXITO: Guardar Captura WebP, Registrar en BD y Retornar "reconocido"
```

---

## 🔌 Catálogo de Endpoints de la API

Todos los endpoints transaccionales requieren el encabezado HTTP:
```http
X-API-Key: tu_clave_secreta
```

### 1. `POST /register-persona`
Busca una persona por CI. Si no existe, la crea en la tabla `persona`.

- **Headers:** `X-API-Key: <API_KEY>`
- **Content-Type:** `application/x-www-form-urlencoded`
- **Parámetros Form:**
  - `ci` *(string, obligatorio)*: Cédula de identidad.

**Ejemplo de respuesta (200 OK):**
```json
{
  "success": true,
  "persona_id": 5
}
```

---

### 2. `POST /register`
Registra un nuevo embedding facial individual para una persona previamente creada.

- **Headers:** `X-API-Key: <API_KEY>`
- **Content-Type:** `multipart/form-data`
- **Parámetros Form:**
  - `persona_id` *(int, obligatorio)*: ID de la persona en la base de datos.
  - `ci` *(string, obligatorio)*: CI correspondiente.
  - `posicion` *(string, obligatorio)*: Una de: `centro`, `izquierda`, `derecha`, `arriba`, `abajo`, `sonrisa`.
  - `image` *(file, obligatorio)*: Imagen JPEG/PNG (máx 5MB).

**Validaciones aplicadas:**
- La persona debe existir y el CI coincidir.
- No superar el tope de 80 embeddings por persona.
- Exactamente 1 rostro visible.
- Anti-spoofing liveness aprobado.
- No debe portar lentes/gafas.

**Ejemplo de respuesta exitosa (200 OK):**
```json
{
  "success": true,
  "message": "Embedding registrado.",
  "total_embeddings": 35,
  "faltan": 35,
  "tiempo_proceso_ms": 115
}
```

**Ejemplo de rechazo por lentes (200 OK):**
```json
{
  "match": false,
  "resultado": "desconocido",
  "message": "Por favor, quítese las gafas para la verificación.",
  "eyeglass_detected": true
}
```

---

### 3. `POST /verify`
Verificación de identidad biométrica en tiempo real mediante dos imágenes simultáneas (frontal y con gesto dinámico).

- **Headers:** `X-API-Key: <API_KEY>`
- **Content-Type:** `multipart/form-data`
- **Parámetros Form:**
  - `persona_id` *(int, obligatorio)*: ID de la persona a verificar.
  - `ci` *(string, obligatorio)*: Cédula de identidad.
  - `gesto_solicitado` *(string, obligatorio)*: Uno de: `arriba`, `abajo`, `izquierda`, `derecha`, `sonrisa`.
  - `tipo_marcado` *(string, opcional)*: Tipo de evento (`"entrada"`, `"salida"` o `"verificando"`, por defecto `"verificando"`).
  - `foto_frontal` *(file, obligatorio)*: Foto tomada de frente sin gafas.
  - `foto_gesto` *(file, obligatorio)*: Foto ejecutando el gesto solicitado.

**Respuestas Posibles:**

#### A. Verificación Exitosa (`reconocido`)
```json
{
  "match": true,
  "resultado": "reconocido",
  "confianza": 84.52,
  "tiempo_proceso_ms": 142,
  "imagen_captura": "9899401/20-10-2026_Martes/entrada_reconocido_08h-30m-15s.webp",
  "log_id": 512
}
```

#### B. Suplantación Detectada (`spoofing_detectado`)
```json
{
  "match": false,
  "resultado": "spoofing_detectado",
  "message": "Posible suplantación detectada.",
  "liveness_score": 0.0821,
  "log_id": 513
}
```

#### C. Gesto No Coincide (`gesto_no_coincide`)
```json
{
  "match": false,
  "resultado": "gesto_no_coincide",
  "message": "El gesto 'izquierda' no coincide. Por favor, realice el gesto solicitado.",
  "gesture_detected": false,
  "log_id": 514
}
```

#### D. Presencia de Lentes (`desconocido` con flag)
```json
{
  "match": false,
  "resultado": "desconocido",
  "message": "Por favor, quítese las gafas para la verificación.",
  "eyeglass_detected": true
}
```

#### E. No Reconocido / No Coincide Identidad (`desconocido`)
```json
{
  "match": false,
  "resultado": "desconocido",
  "confianza": 24.10,
  "message": "No se pudo verificar la identidad.",
  "tiempo_proceso_ms": 138,
  "log_id": 515
}
```

---

### 4. `GET /status/{persona_id}`
Consulta el estado de habilitación biométrica de una persona.

- **Headers:** `X-API-Key: <API_KEY>`
- **Parámetros URL:** `persona_id` (int)

**Ejemplo de respuesta (200 OK):**
```json
{
  "persona_id": 5,
  "activo": true,
  "total_embeddings": 76,
  "calidad_promedio": 0.884,
  "habilitado": true
}
```
> **Nota:** `habilitado` es `true` únicamente si `total_embeddings >= 70`.

---

### 5. `GET /health`
Endpoint de comprobación de salud del servicio (público, no requiere API Key).

**Ejemplo de respuesta (200 OK):**
```json
{
  "status": "ok",
  "service": "reconocimiento-facial"
}
```

---

## 🖥️ Herramienta de Registro Facial (GUI)

El proyecto incluye una aplicación gráfica en Tkinter: `register_ui.py`, diseñada para guiar al usuario en el enrolamiento completo de 76 fotografías:

```bash
python register_ui.py
```

### Distribución de Capturas:
- **Centro:** 10 fotos frontales.
- **Izquierda:** 14 fotos con rotación izquierda.
- **Derecha:** 14 fotos con rotación derecha.
- **Arriba:** 14 fotos mirando hacia arriba.
- **Abajo:** 14 fotos mirando hacia abajo.
- **Sonrisa:** 10 fotos sonriendo.
- **Total:** 76 capturas (supera el mínimo requerido de 70 para habilitar al usuario).

La interfaz muestra la vista en vivo de la cámara, la posición solicitada, barra de progreso interactiva y envía los frames directamente a `POST /register`.

---

## 📸 Almacenamiento de Capturas y Evidencias

El componente `capture_manager.py` almacena automáticamente las fotos de verificación para propósitos de auditoría legal y control:

- **Formato:** WebP con calidad 80 (equilibrio óptimo entre fidelidad y tamaño en disco, aprox. 15-30 KB por imagen).
- **Estructura jerárquica en disco:**
  ```text
  capturas/
  └── {CI}/
      └── {dd-mm-aaaa_Día}/
          └── {tipo_marcado}_{resultado}_{HHh-MMm-SSs}.webp
  ```
  *Ejemplo:* `capturas/9899401/20-10-2026_Martes/entrada_reconocido_08h-30m-15s.webp`

- **Nombres de archivo según resultado:**
  - `reconocido`
  - `desconocido`
  - `gesto_no_coincide`
  - `spoofing`
  - `con_lentes`
  - `error`

---

## 📝 Sistema de Logs y Auditoría

1. **Log Diario de Verificaciones:**
   - Ubicación: `logs/verify-YYYY-MM-DD.log`
   - Formato compacto de 2 líneas por intento:
     ```text
     2026-10-20 08:30:14,980 [VERIFY] persona_id=5 | ci=9899401 | gesto=izquierda | tipo=entrada
     2026-10-20 08:30:15,122 [RESULTADO] reconocido | conf=84.52% | log_id=512 | imagen=9899401/... | 142ms
     ```

2. **Auditoría en Base de Datos:**
   - Cada intento se guarda en la tabla `log_reconocimiento` con el `tiempo_proceso_ms`, IP de origen, score de liveness y enlace a la imagen WebP.

---

## 🔒 Buenas Prácticas y Producción

1. **Gestión de Secretos:** No subir el archivo `.env` al repositorio de código. Usar una clave de API robusta y rotarla periódicamente.
2. **Reverse Proxy:** Desplegar el servicio detrás de Nginx o Traefik configurando terminación SSL/TLS (HTTPS) y limitación de tasa (rate limiting).
3. **Mantenimiento de Imágenes:** Configurar una tarea programada (cron) para archivar o purgar capturas de evidencias con más de 90 días si el almacenamiento es limitado.
4. **Pool de Base de Datos:** `database.py` utiliza `ThreadedConnectionPool` (2 a 10 conexiones concurrentes). En entornos con alta concurrencia, ajustar `minconn` y `maxconn` según los workers de Uvicorn.
5. **Detección de Lentes:** Si el servidor no dispone de GPU, el modelo de lentes corre eficientemente en CPU gracias a la versión `small` y el sistema de caché en memoria de 5 segundos.

---

## 📄 Licencia

Sistema propietario — Uso restringido exclusivamente al Sistema de Asistencia de la institución.