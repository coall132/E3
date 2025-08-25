import os
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

os.environ.setdefault("DISABLE_WARMUP", "1")

from API import main
from API import models
from API.database import Base

def test_auth_flow_and_predict(client):  # <-- utilise le fixture client
    # Prépare le catalogue
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

    # 1) créer une API key
    payload = {"email": "u@test", "username": "user1"}
    r = client.post("/auth/api-keys", params={"password": "coall"}, json=payload)
    assert r.status_code == 200, r.text
    api_key = r.json()["api_key"]

    # 2) obtenir un token
    r = client.post("/auth/token", headers={"X-API-KEY": api_key})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]

    # 3) predict
    form = {"price_level": 2, "city": "37000", "open": "ouvert_lundi_midi",
            "options": ["delivery"], "description": "pizza"}
    r = client.post("/predict?k=3", json=form, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["k"] == 3
    assert len(body["items"]) == 3