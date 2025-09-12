from fastapi import FastAPI, Depends, HTTPException, Security, status, Query, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, selectinload
from typing import List, Optional
import os, time
from datetime import timedelta, datetime, timezone
import pandas as pd
import numpy as np
from joblib import load
from sqlalchemy import MetaData, Table, select, outerjoin
import joblib
from pathlib import Path
import mlflow, os, uuid, json, time
from sqlalchemy.exc import IntegrityError
import logging
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI") 
MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT", "reco-inference")
PROXY_K_INFER = 2          
DIFF_SCALE = 0.05 

try :
    from . import utils
    from . import CRUD
    from . import models
    from . import schema
    from .database import engine, get_db, SessionLocal
    from .features import(
        score_func,
        pair_features,
        form_to_row,
        build_item_features_df,
        aggregate_gains,
        W_eval,
        pick_anchors_from_df,
        text_features01)
except :
    from API import utils
    from API import CRUD
    from API import models
    from API import schema
    from API.database import engine, get_db, SessionLocal
    from . import features as fx
    score_func = fx.score_func
    pair_features = fx.pair_features
    form_to_row = fx.form_to_row
    build_item_features_df = fx.build_item_features_df
    aggregate_gains = fx.aggregate_gains
    W_eval = fx.W_eval
    pick_anchors_from_df = fx.pick_anchors_from_df
    text_features01 = fx.text_features01 

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

def _safe_db_bootstrap(engine):
    # Skip complet en CI / e2e
    if os.getenv("DISABLE_DB_INIT", "0") == "1":
        print("[startup] DB init disabled (DISABLE_DB_INIT=1)")
        return

    dialect = engine.dialect.name

    if dialect in ("postgresql", "postgres"):
        # Chemin Postgres : schéma + tables + éventuels foreign tables
        models.ensure_ml_schema(engine)
        if getattr(models, "_attach_external_tables", None):
            try:
                models._attach_external_tables(engine)
            except Exception as e:
                print(f"[startup] _attach_external_tables skipped: {e}")
        models.Base.metadata.create_all(bind=engine, checkfirst=True)

    elif dialect == "sqlite":
        from sqlalchemy import create_engine
        with engine.begin() as conn:
            conn = conn.execution_options(schema_translate_map={"ml": None,"user_base":None})
            models.Base.metadata.create_all(bind=conn, checkfirst=True)

    else:
        models.Base.metadata.create_all(bind=engine, checkfirst=True)

_safe_db_bootstrap(engine)

@app.on_event("startup")
def _init_db_schema():
    if os.getenv("DISABLE_DB_INIT", "0") == "1":
        print("[startup] DB init disabled (DISABLE_DB_INIT=1).")
        return
    try:
        models.ensure_ml_schema(engine)
        models._attach_external_tables(engine)
        models.Base.metadata.create_all(bind=engine)
        print("[startup] DB schema ensured.")
    except Exception as e:
        print(f"[startup] DB init failed/skipped: {e}")

@app.on_event("startup")
def warmup():
    # Mode dev: ne rien charger de lourd
    if os.getenv("DISABLE_WARMUP", "0") == "1":
        app.state.DF_CATALOG    = pd.DataFrame()
        app.state.SENT_MODEL    = utils._StubSentModel()  # stub léger
        app.state.PREPROC       = None
        app.state.ML_MODEL      = None
        app.state.X_ITEMS       = None
        app.state.FEATURE_COLS  = []
        app.state.ANCHORS       = None
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
        print("[startup] Warmup désactivé (DISABLE_WARMUP=1).")
        return

    # 1) Catalogue
    try:
        df = CRUD.load_df()
        if df is None or df.empty:
            raise RuntimeError("DF_CATALOG vide")
        app.state.DF_CATALOG = df
    except Exception as e:
        print(f"[startup] Échec chargement DF_CATALOG: {e}")
        app.state.DF_CATALOG = pd.DataFrame()

    # 2) Artifacts ML (préproc fitté + rank model + modèle d’embed)
    try:
        ml = CRUD.load_ML()
        app.state.ML_MODEL      = getattr(ml, "rank_model", None)
        app.state.PREPROC       = getattr(ml, "preproc", None)
        app.state.SENT_MODEL    = getattr(ml, "sent_model", None)
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION") or getattr(ml, "rank_model_path", None) or "dev"

        if app.state.SENT_MODEL is None:
            # stub dimensionné d’après les embeddings présents dans le DF
            dim = utils._infer_embed_dim(app.state.DF_CATALOG, default=1024)
            app.state.SENT_MODEL = utils._StubSentModel(dim=dim)

        print(f"[startup] ML: PREPROC={'ok' if app.state.PREPROC is not None else 'None'} | "
              f"MODEL={'ok' if app.state.ML_MODEL is not None else 'None'}")
    except Exception as e:
        print(f"[startup] Échec chargement ML: {e}")
        app.state.ML_MODEL      = None
        app.state.PREPROC       = None
        # fallback texte: stub
        dim = utils._infer_embed_dim(app.state.DF_CATALOG, default=1024)
        app.state.SENT_MODEL    = utils._StubSentModel(dim=dim)
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

    df = app.state.DF_CATALOG

    # 3) Ancres pour features texte (optionnel)
    app.state.ANCHORS = pick_anchors_from_df(df, n=8) if not df.empty else None

    # 4) Pré-calcul X_ITEMS avec le préproc **fitté** (transform, pas fit_transform)
    app.state.X_ITEMS = None
    if app.state.PREPROC is not None and not df.empty:
        try:
            X_items_sp = app.state.PREPROC.transform(df)
            X_items = X_items_sp.toarray().astype(np.float32) if hasattr(X_items_sp, "toarray") \
                      else np.asarray(X_items_sp, dtype=np.float32)
            app.state.X_ITEMS = X_items
            print(f"[startup] X_ITEMS ready, shape={X_items.shape}")
        except Exception as e:
            print(f"[startup] Impossible de construire X_ITEMS: {e}")
            app.state.X_ITEMS = None

    app.state.FEATURE_COLS = []

    nfi = getattr(app.state.ML_MODEL, "n_features_in_", None)
    if nfi is not None and app.state.X_ITEMS is not None:
        expected = int(app.state.X_ITEMS.shape[1]) + 2  # +2 = [cos_desc01, cos_rev01]
        if nfi != expected:
            print(
                f"[startup][warn] Le modèle attend n_features_in_={nfi} mais pair_features produira {expected}. "
                f"Vérifie que rank_model.joblib et preproc_items.joblib proviennent du même train/catalogue."
            )
    anch_shape = None if app.state.ANCHORS is None else getattr(app.state.ANCHORS, "shape", None)
    print(f"[startup] OK | rows={len(df)} | X_ITEMS={(None if app.state.X_ITEMS is None else app.state.X_ITEMS.shape)} "
          f"| anchors={anch_shape}")
    
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
def predict(form: schema.Form,k: int = 3,use_ml: bool = True,user_id: int = Depends(CRUD.current_user_id),db: Session = Depends(get_db)):
    t0 = time.perf_counter()

    if (not hasattr(app.state, "DF_CATALOG")) or app.state.DF_CATALOG is None or app.state.DF_CATALOG.empty:
        raise HTTPException(500, "Catalogue vide/non chargé.")
    df = app.state.DF_CATALOG

    try:
        form_row = models.FormDB(
            price_level=form.price_level,
            city=form.city,      
            open=form.open,
            options=form.options,
            description=form.description,
        )
        db.add(form_row)
        db.flush()
        form_id = form_row.id
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Insertion du formulaire impossible: {e}")

    anchors = getattr(app.state, "ANCHORS", None)
    X_df, gains_proxy = build_item_features_df(df=df,form=form.model_dump(),sent_model=app.state.SENT_MODEL,include_query_consts=True,anchors=anchors,)

    used_ml = False
    scores = np.asarray(gains_proxy, dtype=np.float32)
    model  = getattr(app.state, "ML_MODEL", None)
    preproc = getattr(app.state, "PREPROC", None)
    X_items = getattr(app.state, "X_ITEMS", None)

    if use_ml and (model is not None) and (preproc is not None) and (X_items is not None):
        try:
            Zf_sp = preproc.transform(form_to_row(form.model_dump(), df))
            Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

            T_feat = text_features01(df, form.model_dump(), app.state.SENT_MODEL, k=PROXY_K_INFER)

            Xq = pair_features(Zf, X_items, T_feat, diff_scale=DIFF_SCALE)

            scores = utils._predict_scores(model, Xq)
            used_ml = True

        except Exception as e:
            print(f"[predict] chemin ML en échec, fallback proxy: {e}")
            used_ml = False

    k = int(max(1, min(k or 10, 50)))
    order = np.argsort(scores)[::-1]
    sel = order[:k]

    latency_ms = int((time.perf_counter() - t0) * 1000)
    model_version = os.getenv("MODEL_VERSION") or getattr(app.state, "MODEL_VERSION", None) or "dev"

    pred_row = models.Prediction(
        form_id=form_id,
        k=k,
        model_version=model_version,
        latency_ms=latency_ms,
        status="ok"
    )
    if hasattr(models.Prediction, "user_id"):
        setattr(pred_row, "user_id", user_id)

    items = []
    for r, i in enumerate(sel, start=1):
        etab_id = int(df.iloc[i]["id_etab"]) if "id_etab" in df.columns else int(i)
        items.append(models.PredictionItem(rank=r, etab_id=etab_id, score=float(scores[i])))

    # ---------- Persistance robuste ----------
    try:
        db.add(pred_row)
        db.flush() 

        CRUD.ensure_etabs_exist(db, [it.etab_id for it in items])

        pred_row.items = items
        db.flush()
        db.commit()
        db.refresh(pred_row)

    except IntegrityError as e:
        db.rollback()
        logger.warning("FK violation lors de l'insertion des items: %s. "
                    "On sauvegarde la prédiction sans items.", e)

        try:
            pred_row.items = []
            db.add(pred_row)
            db.flush()
            db.commit()
            db.refresh(pred_row)
        except Exception as e2:
            db.rollback()
            raise HTTPException(500, f"Insertion de la prédiction impossible: {e2}")

    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Insertion de la prédiction impossible: {e}")

    try:
        pyd_pred = schema.Prediction.model_validate(pred_row)
        CRUD.log_prediction_event(
            prediction=pyd_pred,
            form_dict=form.model_dump(),
            scores=np.asarray(scores, dtype=float),
            used_ml=used_ml,
            latency_ms=latency_ms,
            model_version=model_version,
        )
    except Exception as e:
        print(f"[mlflow] log_prediction_event failed: {e}")

    base = schema.Prediction.model_validate(pred_row).model_dump()
    pred_id = str(pred_row.id)
    base.setdefault("id", pred_id)
    base["prediction_id"] = pred_id

    ids = [int(it["etab_id"]) for it in base.get("items", [])]
    details_map = CRUD.get_etablissements_details_bulk(db, ids)

    items_rich = []
    for it in base.get("items", []):
        d = details_map.get(int(it["etab_id"]))
        items_rich.append({**it, "details": d})

    base["items_rich"] = items_rich
    base["message"] = "N’hésitez pas à donner un feedback (0 à 5) via /feedback en utilisant prediction_id."
    return base



@app.post("/feedback", response_model=schema.FeedbackOut, tags=["monitoring"])
def submit_feedback(payload: schema.FeedbackIn,sub: str = Depends(CRUD.get_current_subject),
                    db: Session = Depends(get_db),user_id: int = Depends(CRUD.current_user_id)):
    pred = db.query(models.Prediction).options(selectinload(models.Prediction.items)).filter(models.Prediction.id == payload.prediction_id).first()
    if not pred:
        raise HTTPException(404, "Prediction introuvable")
    
    if getattr(pred, "user_id", None) not in (None, user_id):
        raise HTTPException(status_code=403, detail="Cette prédiction n'appartient pas à l'utilisateur courant")
    

    row = models.Feedback(prediction_id=pred.id,rating=payload.rating,comment=payload.comment)
    db.add(row); db.commit()

    try:
        CRUD.log_feedback_rating(prediction_id=str(pred.id),rating=payload.rating,k=pred.k,model_version=pred.model_version,
        user_id=user_id,comment=payload.comment,use_active_run_if_any=True)
    except Exception as e:
        print(f"[mlflow] log_feedback_rating failed: {e}")

    return schema.FeedbackOut()