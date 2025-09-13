#!/usr/bin/env sh
set -e

if [ -n "$POSTGRES_HOST" ] && [ -n "$POSTGRES_USER" ] && [ -n "$POSTGRES_DB" ]; then
  echo "Waiting for Postgres at ${POSTGRES_HOST}:${POSTGRES_PORT:-5432}..."
  until psql "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT:-5432}/${POSTGRES_DB}?sslmode=disable" -c 'SELECT 1' >/dev/null 2>&1; do
    sleep 2
  done
fi

exec gunicorn --bind 0.0.0.0:5000 --workers 4 mlflow.server:app
