# monitor/monitor.py
import os
import time
from datetime import datetime, timezone
import requests
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

RETRY_SEC = int(os.getenv("MLFLOW_RETRY_SEC", "5"))
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

def wait_for_mlflow(uri: str, experiment: str) -> tuple[MlflowClient, str]:
    """Boucle jusqu'à ce que MLflow réponde; crée l'expérience si besoin."""
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    while True:
        try:
            exp = client.get_experiment_by_name(experiment)
            if exp is None:
                exp_id = client.create_experiment(experiment)
                print(f"[monitor] experiment '{experiment}' created id={exp_id}", flush=True)
            else:
                exp_id = exp.experiment_id
            print(f"[monitor] MLflow OK at {uri} | experiment_id={exp_id}", flush=True)
            return client, exp_id
        except Exception as e:
            print(f"[monitor] MLflow unreachable at {uri}: {e} — retry in {RETRY_SEC}s", flush=True)
            time.sleep(RETRY_SEC)

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

def fetch_last_n_metric(client: MlflowClient, exp_id: str, stage_tag: str | None, metric_name: str, n: int):
    # On récupère "un peu plus" de runs récents et on filtre en Python
    runs = client.search_runs(
        [exp_id],
        order_by=["attributes.start_time DESC"],
        max_results=max(n * 3, 50),
    )
    vals = []
    for r in runs:
        if stage_tag and r.data.tags.get("stage") != stage_tag:
            continue
        m = r.data.metrics.get(metric_name)
        if m is not None:
            vals.append(m)
            if len(vals) >= n:
                break
    return vals

def log_monitor_metrics_to_mlflow(exp_id: str, lat_ma10: float | None, rating_ma10: float | None):
    """Optionnel : loguer les moyennes glissantes dans un run 'monitor' pour les grapher dans MLflow."""
    tags = {"stage": "monitor", "endpoint": "/monitor"}
    with mlflow.start_run(run_name="monitor-tick", experiment_id=exp_id):
        if lat_ma10 is not None:
            mlflow.log_metric("latency_ma10_ms", float(lat_ma10))
        if rating_ma10 is not None:
            mlflow.log_metric("user_rating_ma10", float(rating_ma10))

def main_loop():
    print(f"[monitor] boot with MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI} exp={MLFLOW_EXPERIMENT}", flush=True)
    client, exp_id = wait_for_mlflow(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)

    while True:
        try:
            lat_vals = fetch_last_n_metric(client, exp_id, stage_tag="inference", metric_name="latency_ms", n=LAT_AVG_N)

            lat_ma = float(np.mean(lat_vals)) if lat_vals else None

            rating_vals = fetch_last_n_metric(client, exp_id, stage_tag="feedback", metric_name="user_rating", n=RATING_AVG_N)
            
            rating_ma = float(np.mean(rating_vals)) if rating_vals else None

            log_monitor_metrics_to_mlflow(exp_id, lat_ma, rating_ma)

            now = time.time()
            if lat_ma is not None and lat_ma > LAT_THRESHOLD_MS:
                if now - _last_alert_sent["latency"] > ALERT_COOLDOWN_SEC:
                    send_discord_alert("🚨 Latence élevée",
                        f"Latence moyenne des {LAT_AVG_N} dernières requêtes = {lat_ma:.0f} ms (> {LAT_THRESHOLD_MS} ms).",
                        {"MLflow exp": MLFLOW_EXPERIMENT})
                    _last_alert_sent["latency"] = now

            if rating_ma is not None and rating_ma < RATING_MIN_THRESHOLD:
                if now - _last_alert_sent["rating"] > ALERT_COOLDOWN_SEC:
                    send_discord_alert("🚨 Satisfaction en baisse",
                        f"Note moyenne des {RATING_AVG_N} derniers feedbacks = {rating_ma:.2f} (< {RATING_MIN_THRESHOLD}).",
                        {"MLflow exp": MLFLOW_EXPERIMENT})
                    _last_alert_sent["rating"] = now

            print(f"[monitor] lat_vals={lat_vals} -> MA={lat_ma}", flush=True)
            print(f"[monitor] rating_vals={rating_vals} -> MA={rating_ma}", flush=True)
            print("[monitor] tick OK", flush=True)

        except Exception as e:
            print(f"[monitor] error: {e}", flush=True)

        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main_loop()
