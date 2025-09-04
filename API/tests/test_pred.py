import os
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

os.environ.setdefault("DISABLE_WARMUP", "1")
API_STATIC_KEY=os.getenv("API_STATIC_KEY")

from API import main
from API import models
from API.database import Base

def test_auth_flow_and_predict(client_realdb):
    payload = {"email": "u@test", "username": "user1"}
    r = client_realdb.post("/auth/api-keys", params={"password": API_STATIC_KEY}, json=payload)
    assert r.status_code == 200, r.text
    api_key = r.json()["api_key"]

    r = client_realdb.post("/auth/token", headers={"X-API-KEY": api_key})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]

    form = {"price_level": 2, "code_postal": "37000", "open": "ouvert_lundi_midi",
            "options": ["delivery"], "description": "pizza"}
    r = client_realdb.post("/predict?k=3", json=form,
                           headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["k"] == 3
    assert len(body["items"]) == 3

def test_feedback_good(client_realdb):
    n = 4
    df = pd.DataFrame({
        "id_etab": range(1, n+1),
        "rating": [4.0, 3.0, 5.0, 4.5],
        "priceLevel": [1, 2, 3, 2],
        "latitude": [47.39]*n,
        "longitude": [0.68]*n,
        "editorialSummary_text": ["a", "b", "c", "d"],
        "start_price": [10, 14, 30, 18],
        "code_postal": ["37000", "37000", "37100", "37100"],
        "delivery": [True, False, True, False],
        "servesVegetarianFood": [True, True, False, True],
        "desc_embed": [np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32)]*n,
        "rev_embeds": [None]*n,
        "ouvert_lundi_midi": [1, 1, 0, 1],
    })
    main.app.state.DF_CATALOG = df

    r = client_realdb.post("/auth/api-keys", params={"password": API_STATIC_KEY},
                    json={"email": "uA@test", "username": "userA"})
    assert r.status_code == 200, r.text
    api_key_A = r.json()["api_key"]

    r = client_realdb.post("/auth/token", headers={"X-API-KEY": api_key_A})
    assert r.status_code == 200, r.text
    token_A = r.json()["access_token"]

    form = {"price_level": 2, "city": "37000", "open": "ouvert_lundi_midi",
            "options": ["delivery"], "description": "pizza"}
    r = client_realdb.post("/predict?k=3", json=form, headers={"Authorization": f"Bearer {token_A}"})
    assert r.status_code == 200, r.text
    body = r.json()

    prediction_id = body.get("prediction_id") or body.get("id")
    assert prediction_id is not None

    fb_payload = {"prediction_id": prediction_id, "rating": 4, "comment": "Pertinent"}
    r = client_realdb.post("/feedback", json=fb_payload, )
    assert r.status_code == 401, r.text

    fb_payload = {"prediction_id": prediction_id, "rating": 4, "comment": "Pertinent"}
    r = client_realdb.post("/feedback", json=fb_payload, headers={"Authorization": f"Bearer {token_A}"})
    assert r.status_code == 200, r.text

    r = client_realdb.post("/feedback", json=fb_payload)
    assert r.status_code == 401

def test_feedback_forbidden(client_realdb):
    n = 3
    df = pd.DataFrame({
        "id_etab": range(1, n+1),
        "rating": [4.0, 3.5, 4.2],
        "priceLevel": [1, 2, 2],
        "latitude": [47.39]*n,
        "longitude": [0.68]*n,
        "editorialSummary_text": ["x", "y", "z"],
        "start_price": [10, 12, 20],
        "code_postal": ["37000", "37100", "37200"],
        "delivery": [True, False, True],
        "servesVegetarianFood": [True, False, True],
        "desc_embed": [np.array([0.1, 0.0, 0.2], dtype=np.float32)]*n,
        "rev_embeds": [None]*n,
        "ouvert_lundi_midi": [1, 0, 1],
    })
    main.app.state.DF_CATALOG = df

    r = client_realdb.post("/auth/api-keys", params={"password": API_STATIC_KEY},
                    json={"email": "uA2@test", "username": "userA2"})
    api_key_A = r.json()["api_key"]
    r = client_realdb.post("/auth/token", headers={"X-API-KEY": api_key_A})
    token_A = r.json()["access_token"]

    r = client_realdb.post("/auth/api-keys", params={"password": API_STATIC_KEY},
                    json={"email": "uB@test", "username": "userB"})
    api_key_B = r.json()["api_key"]
    r = client_realdb.post("/auth/token", headers={"X-API-KEY": api_key_B})
    token_B = r.json()["access_token"]

    form = {"price_level": 1, "city": "37000", "open": "ouvert_lundi_midi",
            "options": [], "description": "pâtes"}
    r = client_realdb.post("/predict?k=2", json=form, headers={"Authorization": f"Bearer {token_A}"})
    assert r.status_code == 200, r.text
    prediction_id = r.json().get("prediction_id")
    assert prediction_id is not None

    fb_payload = {"prediction_id": prediction_id, "rating": 1, "comment": "Nope"}
    r = client_realdb.post("/feedback", json=fb_payload, headers={"Authorization": f"Bearer {token_B}"})
    assert r.status_code == 403

    fb_payload = {"prediction_id": prediction_id, "rating": 1, "comment": "Nope"}
    r = client_realdb.post("/feedback", json=fb_payload, headers={"Authorization": f"Bearer {token_A}"})
    assert r.status_code == 200