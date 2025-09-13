FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp

RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "mlflow==2.14.0" gunicorn psycopg2-binary

# Script d’entrée: attend Postgres puis lance gunicorn (même logique que ton compose actuel)
WORKDIR /srv/mlflow
COPY docker/entrypoint-mlflow.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 5000
CMD ["/entrypoint.sh"]
