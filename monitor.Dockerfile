FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# Copie le code du monitor
WORKDIR /app
COPY API/monitor /app/API/monitor

# Dépendances
RUN pip install --no-cache-dir "mlflow==2.14.0" requests numpy

# Script d’entrée: attend que MLflow réponde puis lance le monitor
COPY docker/entrypoint-monitor.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /app/API
CMD ["/entrypoint.sh"]
