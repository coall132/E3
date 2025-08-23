from fastapi import FastAPI, Depends, HTTPException, Security, status, Query
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os, time
from datetime import timedelta, datetime, timezone

from . import CRUD
from . import models
from . import schema
from .database import engine, get_db

app = FastAPI(
    title="API Reco Restaurant",
    description="API pour recommander des restaurants",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_STATIC_KEY = os.getenv("API_STATIC_KEY", "coall")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API des restaurants. Allez sur /docs pour voir les endpoints."}

@app.post("/auth/api-keys", response_model=schema.ApiKeyResponse, tags=["Auth"])
def create_api_key(API_key: schema.ApiKeyCreate,password:str, db: Session = Depends(get_db)):
    if password != API_STATIC_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password invalide ou manquante.",
            headers={"WWW-Authenticate": "APIKey"},
        )
    user = db.query(models.User).filter(models.User.email == API_key.email).first()
    if user is None:
        if db.query(models.User).filter(models.User.username == API_key.username).first():
            raise HTTPException(status_code=409, detail="Ce username est déjà pris.")
        user = models.User(email=API_key.email, username=API_key.username)
        db.add(user)
        db.flush()

    api_key_plain, key_id, _secret = CRUD.generate_api_key()
    key_hash = CRUD.hash_api_key(api_key_plain)

    row = models.ApiKey(user_id=user.id,key_id=key_id,key_hash=key_hash,name=API_key.name or None,is_active=True,)
    db.add(row)
    db.commit()
    return schema.ApiKeyResponse(api_key=api_key_plain, key_id=key_id)

def verify_api_key(db: Session, api_key: str):
    if not api_key or "." not in api_key or not api_key.startswith("rk_"):
        raise HTTPException(status_code=401, detail="Clé API manquante ou invalide.", headers={"WWW-Authenticate":"APIKey"})

    prefix, _, _secret = api_key.partition(".")
    key_id = prefix.replace("rk_", "", 1)

    row = db.query(models.ApiKey).filter(models.ApiKey.key_id == key_id,).first()

    if not row or not CRUD.verify_api_key_hash(api_key, row.key_hash):
        raise HTTPException(status_code=401, detail="Clé API invalide.", headers={"WWW-Authenticate":"APIKey"})

    row.last_used_at = datetime.now(timezone.utc)
    db.add(row); db.commit()

    return row

@app.post("/auth/token", response_model=schema.TokenOut, tags=["Auth"])
def issue_token(api_key: Optional[str] = Security(api_key_header), db: Session = Depends(get_db)):
    row = verify_api_key(db, api_key)
    token, exp_ts = CRUD.create_access_token(subject=f"user:{row.user_id}")
    return schema.TokenOut(access_token=token, expires_at=exp_ts)