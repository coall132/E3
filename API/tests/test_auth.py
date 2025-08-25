# tests/test_auth.py
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from fastapi.testclient import TestClient
import pytest

def test_create_api_key_success(client): 
    info = {
        "email": "alice@example.com",
        "username": "alice",
        "name": "clé de test"
    }
    r = client.post("/auth/api-keys?password=coall", json=info)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "api_key" in data and data["api_key"].startswith("rk_")
    assert "key_id" in data and len(data["key_id"]) > 0

def test_create_api_key_bad_password(client):
    info = {"email": "bob@example.com", "username": "bob"}
    r = client.post("/auth/api-keys?password=wrong", json=info)
    assert r.status_code == 401
    assert "Password invalide" in r.text

def test_issue_token_with_api_key(client):
    info = {"email": "carol@example.com", "username": "carol"}
    r = client.post("/auth/api-keys?password=coall", json=info)
    assert r.status_code == 200
    api_key = r.json()["api_key"]

    r2 = client.post("/auth/token", headers={"X-API-KEY": api_key})
    assert r2.status_code == 200, r2.text
    tok = r2.json()
    assert "access_token" in tok and tok["access_token"]
    assert isinstance(tok["expires_at"], int)

def test_issue_token_invalid_key(client):
    r = client.post("/auth/token", headers={"X-API-KEY": "rk_deadbeef.notreal"})
    assert r.status_code == 401
    assert "Clé API invalide" in r.text
