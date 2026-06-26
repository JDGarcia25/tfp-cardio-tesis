# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar TF / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copiar solo archivos de dependencias primero (cache de capas)
COPY pyproject.toml poetry.lock ./

# Exportar requirements sin dev ni hashes
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

# Instalar en directorio separado
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Librerías de sistema en runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes instalados
COPY --from=builder /install /usr/local

# Copiar código fuente
COPY src/ ./src/
COPY models/ ./models/

# Variables de entorno
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_DIR=/app/models \
    CORS_ORIGINS=* \
    RATE_LIMIT_RPM=60 \
    PORT=8000

# Usuario sin privilegios
RUN useradd --no-create-home --shell /bin/false ecg
USER ecg

EXPOSE 8000

# Health check interno
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "ecg_anomaly.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
