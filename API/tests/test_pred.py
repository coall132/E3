import json
import pytest
from sqlalchemy import select, func
from API import models

def _create_api_key(client, email="alice@example.com", username="alice", password="coall", name="clé de test"):
    payload = {"email": email, "username": username, "name": name}
    return client.post("/auth/api-keys", params={"password": password}, json=payload)

def _exchange_token(client, api_key: str):
    return client.post("/auth/token", headers={"X-API-KEY": api_key})

def _auth_headers(client):
    r = _create_api_key(client)
    assert r.status_code == 200, r.text
    api_key = r.json()["api_key"]
    r2 = _exchange_token(client, api_key)
    assert r2.status_code == 200, r2.text
    token = r2.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def _count(db_session, model_cls):
    return db_session.execute(select(func.count()).select_from(model_cls)).scalar_one()

def test_predict_happy_path(client, db_session):
    headers = _auth_headers(client)

    # Compter les lignes avant (si ton /predict persiste en base)
    before_form = _count(db_session, models.FormDB)
    before_pred = _count(db_session, models.Prediction)
    before_item = _count(db_session, models.PredictionItem)

    form = {
        "price_level": 2,
        "code_postal": ["37000", "37100", "37200"],   # adapte si ton schéma diffère
        "open": "ouvert_samedi_soir",
        "options": ["servesVegetarianFood", "outdoorSeating"],
        "description": "italien calme avec terrasse"
    }
    params = {"k": 2}

    r = client.post("/predict", headers=headers, params=params, json=form)
    assert r.status_code == 200, r.text
    data = r.json()

    # Structure de réponse minimale attendue
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) == 2

    for rank, it in enumerate(data["items"], start=1):
        # tolérant sur le nom des clés : etab_id ou id_etab
        assert any(k in it for k in ("etab_id", "id_etab")), it
        assert "score" in it
        # si la réponse contient 'rank', vérifie la cohérence
        if "rank" in it:
            assert it["rank"] == rank

    # Vérifie la persistance (si ton endpoint enregistre form + prediction + items)
    after_form = _count(db_session, models.FormDB)
    after_pred = _count(db_session, models.Prediction)
    after_item = _count(db_session, models.PredictionItem)

    assert after_form == before_form + 1
    assert after_pred == before_pred + 1
    assert after_item >= before_item + 2  # au moins k items

def test_predict_requires_auth(client):
    form = {
        "price_level": 2,
        "code_postal": ["37000"],
        "open": "ouvert_samedi_soir",
        "options": [],
        "description": "bistrot"
    }
    r = client.post("/predict", json=form, params={"k": 2})
    assert r.status_code in (401, 403)

def test_predict_bad_k_validation(client):
    headers = _auth_headers(client)
    form = {
        "price_level": 2,
        "code_postal": ["37000"],
        "open": "ouvert_samedi_soir",
        "options": [],
        "description": "bistrot"
    }
    # k=0 (en dehors de ge=1) -> 422 attendu si tu as la validation Pydantic
    r = client.post("/predict", headers=headers, params={"k": 0}, json=form)
    assert r.status_code in (400, 422)

def test_predict_k_greater_than_catalog_size(client):
    headers = _auth_headers(client)
    form = {
        "price_level": 2,
        "code_postal": ["37000", "37100", "37200"],
        "open": "ouvert_samedi_soir",
        "options": ["servesVegetarianFood"],
        "description": "terrasse"
    }
    r = client.post("/predict", headers=headers, params={"k": 10}, json=form)
    assert r.status_code == 200, r.text
    data = r.json()
    # le catalogue seedé dans la fixture contient 3 restos => on doit avoir <= 3 items
    assert "items" in data and isinstance(data["items"], list)
    assert 1 <= len(data["items"]) <= 3