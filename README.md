# Servicio de Reconocimiento Facial

Microservicio Python (FastAPI) para reconocimiento facial.

## Requisitos
- Python 3.10+
- PostgreSQL 16+

## Instalación

### 1. Clonar repositorio

### 2. Crear entorno virtual
```
python -m venv venv
venv\Scripts\activate # Windows
```

### 3. Instalar dependencias
```
pip install -r requirements.txt
```

### 4. Configurar .env
Copiar `.env.example` a `.env` y configurar:
- DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
- SERVICE_HOST, SERVICE_PORT

### 5. Ejecutar migraciones
```
python migrate.py
```

### 6. Iniciar servicio
```
python main.py
```

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /register-persona | Crear persona por CI |
| POST | /register | Registrar embedding |
| POST | /verify | Verificar rostro |
| GET | /status/{persona_id} | Estado facial |

## Base de datos

- `facial_db.sql` — Script para crear BD en servidores
- `migrate.py` — Migraciones automáticas