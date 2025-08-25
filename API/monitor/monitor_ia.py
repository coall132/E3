# monitor/monitor.py
import os
import time
from datetime import datetime, timezone
import requests
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "restaurant-api")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Règles
LAT_AVG_N = int(os.getenv("LAT_AVG_N", "10"))
LAT_THRESHOLD_MS = int(os.getenv("LAT_THRESHOLD_MS", "10000"))  # 10s

RATING_AVG_N = int(os.getenv("RATING_AVG_N", "10"))
RATING_MIN_THRESHOLD = float(os.getenv("RATING_MIN_THRESHOLD", "1.0"))

CHECK_INTERVAL_SEC = int(os.getenv("CHECK_INTERVAL_SEC", "60"))  # toutes les 60s
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "600")) # anti-spam: 10min

# petit état en mémoire
_last_alert_sent = {"latency": 0, "rating": 0}

def send_discord_alert(title: str, message: str, fields: dict | None = None):
    if not DISCORD_WEBHOOK_URL:
        return
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    if fields:
        payload["embeds"][0]["fields"] = [
            {"name": k, "value": str(v), "inline": True} for k, v in fields.items()
        ]
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=8)
    except Exception as e:
        print(f"[monitor] Discord webhook error: {e}")

def get_or_create_experiment_id(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return client.create_experiment(name)
    return exp.experiment_id

def fetch_last_n_metric(client: MlflowClient, exp_id: str, stage_tag: str, metric_name: str, n: int):
    """Retourne la liste des valeurs de métrique (dernières d’abord) pour les runs taggés stage=..."""
    runs = client.search_runs(
        [exp_id],
        filter_string=f'tags.stage = "{stage_tag}" and metrics.{metric_name} IS NOT NULL',
        order_by=["attribute.start_time DESC"],
        max_results=n
    )
    vals = []
    for r in runs:
        # r.data.metrics est un dict {metric: dernier_val}
        if metric_name in r.data.metrics:
            vals.append(r.data.metrics[metric_name])
    return vals

def log_monitor_metrics_to_mlflow(lat_ma10: float | None, rating_ma10: float | None):
    """Optionnel : loguer les moyennes glissantes dans un run 'monitor' pour les grapher dans MLflow."""
    tags = {"stage": "monitor", "endpoint": "/monitor"}
    with mlflow.start_run(run_name="monitor-tick", nested=True, tags=tags):
        if lat_ma10 is not None:
            mlflow.log_metric("latency_ma10_ms", float(lat_ma10))
        if rating_ma10 is not None:
            mlflow.log_metric("user_rating_ma10", float(rating_ma10))

def main_loop():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp_id = get_or_create_experiment_id(client, MLFLOW_EXPERIMENT)

    while True:
        try:
            # 1) latence moyenne des 10 dernières prédictions
            lat_vals = fetch_last_n_metric(client, exp_id, stage_tag="inference",
                                           metric_name="latency_ms", n=LAT_AVG_N)
            lat_ma = float(np.mean(lat_vals)) if lat_vals else None

            # 2) note moyenne des 10 derniers feedbacks
            rating_vals = fetch_last_n_metric(client, exp_id, stage_tag="feedback",
                                              metric_name="user_rating", n=RATING_AVG_N)
            rating_ma = float(np.mean(rating_vals)) if rating_vals else None

            # (optionnel) loguer ces moyennes dans MLflow pour avoir les courbes MA(10)
            log_monitor_metrics_to_mlflow(lat_ma, rating_ma)

            now = time.time()

            # Règle A: latence > seuil
            if lat_ma is not None and lat_ma > LAT_THRESHOLD_MS:
                if now - _last_alert_sent["latency"] > ALERT_COOLDOWN_SEC:
                    send_discord_alert(
                        "🚨 Latence élevée",
                        f"Latence moyenne des {LAT_AVG_N} dernières requêtes = {lat_ma:.0f} ms (> {LAT_THRESHOLD_MS} ms).",
                        {"MLflow exp": MLFLOW_EXPERIMENT}
                    )
                    _last_alert_sent["latency"] = now

            # Règle B: satisfaction < seuil
            if rating_ma is not None and rating_ma < RATING_MIN_THRESHOLD:
                if now - _last_alert_sent["rating"] > ALERT_COOLDOWN_SEC:
                    send_discord_alert(
                        "🚨 Satisfaction en baisse",
                        f"Note moyenne des {RATING_AVG_N} derniers feedbacks = {rating_ma:.2f} (< {RATING_MIN_THRESHOLD}).",
                        {"MLflow exp": MLFLOW_EXPERIMENT}
                    )
                    _last_alert_sent["rating"] = now

        except Exception as e:
            print(f"[monitor] error: {e}")

        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main_loop()
