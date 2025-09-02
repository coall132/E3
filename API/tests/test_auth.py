# tests/test_auth.py
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from fastapi.testclient import TestClient
import pytest
import os

API_STATIC_KEY=os.getenv("API_STATIC_KEY")

def test_create_api_key_success(client_realdb): 
    info = {
        "email": "alice@example.com",
        "username": "alice",
        "name": "clé de test"
    }
    r = client_realdb.post(f"/auth/api-keys?password={API_STATIC_KEY}", json=info)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "api_key" in data and data["api_key"].startswith("rk_")
    assert "key_id" in data and len(data["key_id"]) > 0

def test_create_api_key_bad_password(client_realdb):
    info = {"email": "bob@example.com", "username": "bob"}
    r = client_realdb.post("/auth/api-keys?password=wrong", json=info)
    assert r.status_code == 401
    assert "Password invalide" in r.text

def test_issue_token_with_api_key(client_realdb):
    info = {"email": "carol@example.com", "username": "carol"}
    r = client_realdb.post(f"/auth/api-keys?password={API_STATIC_KEY}", json=info)
    assert r.status_code == 200
    api_key = r.json()["api_key"]

    r2 = client_realdb.post("/auth/token", headers={"X-API-KEY": api_key})
    assert r2.status_code == 200, r2.text
    tok = r2.json()
    assert "access_token" in tok and tok["access_token"]
    assert isinstance(tok["expires_at"], int)


def test_token_missing_header_returns_401(client_realdb):
    r = client_realdb.post("/predict?k=3", json={"description": "pizza"})
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate", "").startswith("Bearer")

def test_create_api_key_conflict_username(client_realdb):
    p = {"email": "a@x", "username": "userA"}
    r1 = client_realdb.post(f"/auth/api-keys?password={API_STATIC_KEY}", json=p)
    assert r1.status_code == 200
    r2 = client_realdb.post(f"/auth/api-keys?password={API_STATIC_KEY}", json={"email":"b@x", "username":"userA"})
    assert r2.status_code == 409

def test_issue_token_missing_or_malformed_key(client_realdb):
    r = client_realdb.post("/auth/token")
    assert r.status_code == 401
    r2 = client_realdb.post("/auth/token", headers={"X-API-KEY": "not_a_key"})
    assert r2.status_code == 401
