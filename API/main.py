from fastapi import FastAPI, Depends, HTTPException, Security, status, Query, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os, time
from datetime import timedelta, datetime, timezone
import pandas as pd
import numpy as np
from joblib import load
from sqlalchemy import MetaData, Table, select, outerjoin
import joblib
from pathlib import Path

try :
    from . import utils
    from . import CRUD
    from . import models
    from . import schema
    from .database import engine, get_db, SessionLocal
    from . import benchmark_3 as bm
    from .benchmark_2_0 import (
    score_func,
    build_item_features_df,
    aggregate_gains,
    W_eval,  
)
except :
    import utils
    import CRUD
    import models
    import schema
    from database import engine, get_db, SessionLocal
    import benchmark_3 as bm
    from benchmark_2_0 import (
    score_func,
    build_item_features_df,
    aggregate_gains,
    W_eval,  # ou W_proxy si tu préfères
)

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

@app.on_event("startup")
def warmup():
    CRUD, DEFAULT_SENT_MODEL, pick_anchors_from_df, build_item_features_df, make_preproc_final = _late_imports()

    if os.getenv("DISABLE_WARMUP", "0") == "1":
        app.state.DF_CATALOG = pd.DataFrame()
        app.state.SENT_MODEL = DEFAULT_SENT_MODEL
        app.state.PREPROC = None
        app.state.ML_MODEL = None
        app.state.FEATURE_COLS = []
        app.state.ANCHORS = None
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
        print("[startup] Warmup désactivé (DISABLE_WARMUP=1).")
        return

    try:
        df = CRUD.load_df()
        if df is None or df.empty:
            raise RuntimeError("DF_CATALOG vide")
        app.state.DF_CATALOG = df
    except Exception as e:
        print(f"[startup] Échec chargement DF_CATALOG: {e}")
        app.state.DF_CATALOG = pd.DataFrame()

    try:
        ml = CRUD.load_ML()
        app.state.PREPROC = getattr(ml, "preproc", None)
        app.state.PREPROC_FACTORY = getattr(ml, "preproc_factory", None)
        app.state.SENT_MODEL = getattr(ml, "sent_model", None) or DEFAULT_SENT_MODEL
        app.state.ML_MODEL = getattr(ml, "rank_model", None)
        # utile pour tracer la version
        app.state.MODEL_VERSION = (
            os.getenv("MODEL_VERSION")
            or getattr(ml, "rank_model_path", None)
            or "dev"
        )
        print(f"[startup] ML: PREPROC={type(app.state.PREPROC).__name__ if app.state.PREPROC else 'None'} | "
              f"MODEL={'ok' if app.state.ML_MODEL is not None else 'None'}")
    except Exception as e:
        print(f"[startup] Échec chargement ML: {e}")
        app.state.PREPROC = None
        app.state.SENT_MODEL = DEFAULT_SENT_MODEL
        app.state.ML_MODEL = None
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

    df = app.state.DF_CATALOG
    anchors = pick_anchors_from_df(df, n=8) if not df.empty else None
    app.state.ANCHORS = anchors

    feature_cols = []
    if not df.empty:
        neutral_form = {
            "description": "",
            "price_level": np.nan,
            "code_postal": None,
            "options": [],
            "open": "",
        }
        X_probe_df, _ = build_item_features_df(df=df,form=neutral_form,sent_model=app.state.SENT_MODEL,
            include_query_consts=True,anchors=anchors,)
        feature_cols = [c for c in X_probe_df.columns if c != "id_etab"]
        app.state.FEATURE_COLS = feature_cols

        if app.state.PREPROC is None and feature_cols:
            app.state.PREPROC = make_preproc_final().fit(X_probe_df[feature_cols])
            print("[startup] PREPROC fallback créé et fit sur features de probe (DEV ONLY).")
    else:
        app.state.FEATURE_COLS = []

    anch_shape = None if anchors is None else getattr(anchors, "shape", None)
    print(f"[startup] OK | rows={len(df)} | features={len(feature_cols)} | anchors={anch_shape} |")

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API des restaurants. Allez sur /docs pour voir les endpoints."}

@app.post("/auth/api-keys", response_model=schema.ApiKeyResponse, tags=["Auth"])
def create_api_key(API_key_in: schema.ApiKeyCreate,password:str, db: Session = Depends(get_db)):
    if password != API_STATIC_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password invalide ou manquante.",
            headers={"WWW-Authenticate": "APIKey"},
        )
    user = db.query(models.User).filter(models.User.email == API_key_in.email).first()
    if user is None:
        if db.query(models.User).filter(models.User.username == API_key_in.username).first():
            raise HTTPException(status_code=409, detail="Ce username est déjà pris.")
        user = models.User(email=API_key_in.email, username=API_key_in.username)
        db.add(user)
        db.flush()

    api_key_plain, key_id, _secret = CRUD.generate_api_key()
    key_hash = CRUD.hash_api_key(api_key_plain)

    row = models.ApiKey(user_id=user.id,key_id=key_id,key_hash=key_hash or None)
    db.add(row)
    db.commit()
    return schema.ApiKeyResponse(api_key=api_key_plain, key_id=key_id)


@app.post("/auth/token", response_model=schema.TokenOut, tags=["Auth"])
def issue_token(API_key_in: Optional[str] = Security(api_key_header), db: Session = Depends(get_db)):
    row = CRUD.verify_api_key(db, API_key_in)
    token, exp_ts = CRUD.create_access_token(subject=f"user:{row.user_id}")
    return schema.TokenOut(access_token=token, expires_at=exp_ts)

@app.post("/predict", tags=["predict"], dependencies=[Depends(CRUD.get_current_subject)])
def predict(form: schema.Form, k: int = 10, use_ml: bool = True):
    t0 = time.perf_counter()

    if not hasattr(app.state, "DF_CATALOG") or app.state.DF_CATALOG is None or app.state.DF_CATALOG.empty:
        raise HTTPException(500, "Catalogue vide/non chargé.")
    df = app.state.DF_CATALOG

    anchors = getattr(app.state, "ANCHORS", None)
    X_df, gains_proxy = build_item_features_df(df=df,form=form.model_dump(),          sent_model=app.state.SENT_MODEL, 
        include_query_consts=True,anchors=anchors,                  )

    used_ml = False
    scores = gains_proxy.copy()

    if use_ml and getattr(app.state, "ML_MODEL", None) is not None and getattr(app.state, "PREPROC", None) is not None:
        feature_cols = getattr(app.state, "FEATURE_COLS", None)
        if feature_cols is None:
            feature_cols = [c for c in X_df.columns if c != "id_etab"]
            app.state.FEATURE_COLS = feature_cols

        X_df_aligned = utils._align_df_to_cols(X_df.copy(), feature_cols)
        X_sp = app.state.PREPROC.transform(X_df_aligned) 
        X = X_sp.toarray().astype(np.float32) if hasattr(X_sp, "toarray") else np.asarray(X_sp, dtype=np.float32)
        scores = utils._predict_scores(app.state.ML_MODEL, X)
        used_ml = True

    k = int(max(1, min(k, 50)))
    order = np.argsort(scores)[::-1]
    sel = order[:k]

    items: List[schema.PredictionItem] = []
    for r, i in enumerate(sel, start=1):
        etab_id = int(df.iloc[i]["id_etab"]) if "id_etab" in df.columns else int(i)
        items.append(schema.PredictionItem(id=None,prediction_id=None,  rank=r,
                etab_id=etab_id,score=float(scores[i]),))

    latency_ms = int((time.perf_counter() - t0) * 1000)
    model_version = os.getenv("MODEL_VERSION") or getattr(app.state, "MODEL_VERSION", None)

    return schema.Prediction(id=None,form_id=None,k=k,model_version=model_version,latency_ms=latency_ms,
        status="ok",items=items,)
