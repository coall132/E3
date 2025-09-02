FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements_client.txt .
RUN pip install --no-cache-dir -r requirements_client.txt

COPY streamlit.py /app/streamlit.py

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py", "--server.port=8501"]