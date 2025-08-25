FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# requirements dans API/ => chemin relatif au contexte (.)
COPY ./API/requirements_api.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

# copier le code API/ sous /app/API pour garder l'import "API.main:app"
COPY ./API /app/API

EXPOSE 8000
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]