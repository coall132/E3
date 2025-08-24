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

from . import CRUD
from . import models
from . import schema
from .database import engine, get_db, SessionLocal
from . import benchmark_3 as bm

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

app.state.ml_model = None
app.state.sent_model = None
app.state.preproc = None
TOPK_DEFAULT = int(os.getenv("PRED_TOPK_DEFAULT", "3"))

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

def _to_np1d(x):
    # desc_embed : liste -> np.array(float32) ou None
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=np.float32)
        return arr if arr.ndim == 1 else None
    return None

def _to_list_np(x):
    # rev_embeds : liste de vecteurs -> liste de np.array(float32) ou None
    if isinstance(x, list):
        out = []
        for v in x:
            try:
                out.append(np.asarray(v, dtype=np.float32).reshape(-1))
            except Exception:
                pass
        return out if out else None
    return None

@app.on_event("startup")
def warmup_catalog_and_model_sqlalchemy():
    """
    Charge en mémoire:
      - DF_CATALOG (avec desc_embed/rev_embeds transformés)
      - X_ITEMS = PREPROC.fit_transform(DF_CATALOG)
      - ML_MODEL (joblib)
      - PREPROC, SENT_MODEL (depuis benchmark_3)
    Tout est stocké sur app.state.*
    """
    # 1) objets ML de référence
    app.state.PREPROC = getattr(bm, "preproc", None)
    app.state.SENT_MODEL = getattr(bm, "model", None)
    if app.state.PREPROC is None:
        raise RuntimeError("PREPROC introuvable (benchmark_3.preproc).")
    # 2) lire catalogue + embeddings
    metadata = MetaData()
    Emb = Table("etab_embedding", metadata, schema="public", autoload_with=engine)
    Etab = models.Etablissement.__table__  # ta table des restos

    with SessionLocal() as db:
        stmt = (
            select(
                Etab.c.id_etab,
                *[col for col in Etab.c if col.name != "id_etab"],
                Emb.c.desc_embed,
                Emb.c.rev_embeds,
            )
            .select_from(outerjoin(Etab, Emb, Etab.c.id_etab == Emb.c.id_etab))
            .order_by(Etab.c.id_etab)
        )
        rows = db.execute(stmt).mappings().all()

    df = pd.DataFrame(rows)

    # colonnes attendues par benchmark_3 (filets de sécurité)
    for c in getattr(bm, "BOOL_COLS", []):
        if c not in df.columns:
            df[c] = False
    for c in getattr(bm, "NUM_COLS", []):
        if c not in df.columns:
            df[c] = 0.0
    if "priceLevel" not in df.columns: df["priceLevel"] = np.nan
    if "code_postal" not in df.columns: df["code_postal"] = ""
    if "editorialSummary_text" not in df.columns: df["editorialSummary_text"] = ""

    # embeddings -> numpy
    df["desc_embed"] = df.get("desc_embed", None)
    df["rev_embeds"] = df.get("rev_embeds", None)
    if "desc_embed" in df.columns:
        df["desc_embed"] = df["desc_embed"].apply(_to_np1d)
    if "rev_embeds" in df.columns:
        df["rev_embeds"] = df["rev_embeds"].apply(_to_list_np)

    app.state.DF_CATALOG = df

    # 3) features item
    X = app.state.PREPROC.fit_transform(df)
    app.state.X_ITEMS = X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)

    # 4) modèle de ranking
    model_path = os.getenv("RANK_MODEL_PATH", str(Path("artifacts") / "linear_svc_pointwise.joblib"))
    try:
        app.state.ML_MODEL = joblib.load(model_path)
        print(f"[startup] rank model: {model_path}")
    except Exception as e:
        print(f"[startup] impossible de charger {model_path}: {e}")
        # fallback minimal pour pouvoir tourner en dev/tests
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, app.state.X_ITEMS.shape[1])).astype(np.float32)
        y = (rng.random(40) > 0.5).astype(int)
        app.state.ML_MODEL = make_pipeline(StandardScaler(), LinearSVC()).fit(X, y)

@app.post("/predict", tags=["Reco"])
def predict(
    form: schema.Form,
    request: Request,
    k: int = Query(TOPK_DEFAULT, ge=1, le=50),
    _auth: str = Depends(CRUD.get_current_subject),
    db: Session = Depends(get_db),
):
    st = request.app.state
    # Sanity checks
    if any(getattr(st, n, None) is None for n in ("DF_CATALOG","X_ITEMS","ML_MODEL","PREPROC")):
        raise HTTPException(status_code=503, detail="Service ML non prêt.")

    t0 = time.perf_counter()

    # 1) persister le formulaire
    form_db = models.FormDB(
        price_level=form.price_level,
        city=form.city,
        open=form.open,
        options=form.options,
        description=form.description,
    )
    db.add(form_db)
    db.flush()

    # 2) préparer dict compatible avec benchmark_3
    fdict = form.model_dump()
    if form.city:
        try:
            fdict["code_postal"] = bm.city_to_postal_codes_exact(form.city)
        except Exception:
            fdict["code_postal"] = None

    # 3) vecteur requête via ta pipeline
    Zf_sp = st.PREPROC.transform(bm.form_to_row(fdict, st.DF_CATALOG))
    Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

    # 4) signaux H (exactement comme ton fichier)
    H = {
        "price":   bm.h_price_vector_simple(st.DF_CATALOG, fdict),
        "rating":  bm.h_rating_vector(st.DF_CATALOG),
        "city":    bm.h_city_vector(st.DF_CATALOG, fdict),
        "options": bm.h_opts_vector(st.DF_CATALOG, fdict),
        "open":    bm.h_open_vector(st.DF_CATALOG, fdict),
        "text":    bm.score_text(st.DF_CATALOG, fdict, st.SENT_MODEL, w_rev=0.6, w_desc=0.4, k=3, missing_cos=0.0),
    }

    # 5) features requête-item
    Xq = bm.pair_features(Zf, H, st.X_ITEMS)

    # 6) scoring par ton LinearSVC pointwise
    model = st.ML_MODEL
    if hasattr(model, "decision_function"):
        scores = model.decision_function(Xq)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(Xq)[:, 1]
    else:
        scores = model.predict(Xq).astype(float)

    # 7) top-k
    k = int(max(1, min(k, len(scores))))
    order = np.argsort(scores)[::-1][:k]
    top_ids = st.DF_CATALOG.iloc[order]["id_etab"].astype(int).tolist()
    top_scores = [float(scores[i]) for i in order]

    # 8) persistance ml.prediction + ml.prediction_item
    latency_ms = int((time.perf_counter() - t0) * 1000)
    pred = models.Prediction(
        form_id=form_db.id,
        k=k,
        model_version="LinearSVC_pointwise_v1",
        latency_ms=latency_ms,
        status="ok",
    )
    db.add(pred)
    db.flush()

    items = []
    for r, i in enumerate(order, start=1):
        etab_id = int(st.DF_CATALOG.iloc[i]["id_etab"])
        score = float(scores[i])
        db.add(models.PredictionItem(
            prediction_id=pred.id,
            rank=r,
            etab_id=etab_id,
            score=score
        ))
        items.append({"rank": r, "etab_id": etab_id, "score": score})

    db.commit()
    return {"prediction_id": str(pred.id), "k": k, "latency_ms": latency_ms, "items": items}