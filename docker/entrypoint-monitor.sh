#!/usr/bin/env sh
set -e

MLFLOW_URL="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
echo "Waiting for MLflow at ${MLFLOW_URL} ..."
until python - "$MLFLOW_URL" <<'PY'
import sys, urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        sys.exit(0 if 200 <= r.status < 400 else 1)
except Exception:
    sys.exit(1)
PY
do
  echo "waiting mlflow..."
  sleep 2
done

exec python monitor/monitor_ia.py
