from sqlalchemy.orm import Session, joinedload, selectinload
import secrets, base64, time
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import os
from fastapi import FastAPI, Depends, HTTPException, Security, status, Query
from argon2 import PasswordHasher, exceptions as argon_exc
from pathlib import Path
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import pandas as pd
import os, mlflow
from mlflow.tracking import MlflowClient
import io
import math

try:
    from . import models
    from . import database as db
    from . import schema
    from . import benchmark_2_0 as bm
    from . import utils
    from .main import app
    from . import features as fx
except:
    from API import models
    from API import database as db
    from API import schema
    from API import features as fx
    from API import utils

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)  # pour /auth/token uniquement
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
ph = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=8)

API_STATIC_KEY = os.getenv("API_STATIC_KEY")  # pour échanger contre un token
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
_MLFLOW_READY = False
_MLFLOW_EXP_ID = None 

def create_access_token(subject: str, expires_delta: Optional[timedelta] = None):
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"sub": subject, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt, int(expire.timestamp())

async def get_current_subject(token: str = Depends(oauth2_scheme)):
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        sub: Optional[str] = payload.get("sub")
        if sub is None:
            raise credentials_exc
        return sub
    except JWTError:
        raise credentials_exc

def generate_api_key():
    key_id = secrets.token_hex(4)
    secret = base64.urlsafe_b64encode(secrets.token_bytes(24)).decode().rstrip("=")
    api_key_plain = f"rk_{key_id}.{secret}"
    return api_key_plain, key_id, secret

def hash_api_key(api_key_plain: str):
    return ph.hash(api_key_plain)

def verify_api_key_hash(api_key_plain: str, key_hash: str):
    try:
        return ph.verify(key_hash, api_key_plain)
    except argon_exc.VerifyMismatchError:
        return False
    
def verify_api_key(db: Session, API_key_in: str):
    if not API_key_in or "." not in API_key_in or not API_key_in.startswith("rk_"):
        raise HTTPException(status_code=401, detail="Clé API manquante ou invalide.", headers={"WWW-Authenticate":"APIKey"})

    prefix, _, _secret = API_key_in.partition(".")
    key_id = prefix.replace("rk_", "", 1)

    row = db.query(models.ApiKey).filter(models.ApiKey.key_id == key_id,).first()

    if not row or not verify_api_key_hash(API_key_in, row.key_hash):
        raise HTTPException(status_code=401, detail="Clé API invalide.", headers={"WWW-Authenticate":"APIKey"})

    row.last_used_at = datetime.now(timezone.utc)
    db.add(row); db.commit()
    return row

def current_user_id(subject: str = Depends(get_current_subject)) -> int:
    try:
        prefix, uid = subject.split(":", 1)
        if prefix != "user":
            raise ValueError
        return int(uid)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sujet JWT invalide")
    
def load_ML():
    state = schema.MLState()
    state.preproc_factory = None
    try:
        state.sent_model = SentenceTransformer("BAAI/bge-m3")
    except Exception:
        state.sent_model = None

    base = os.getenv("ARTIFACTS_DIR", "artifacts")
    preproc_path = Path(os.getenv("PREPROC_PATH", f"{base}/preproc_items.joblib"))
    rank_path    = Path(os.getenv("RANK_MODEL_PATH", f"{base}/rank_model.joblib"))

    if preproc_path.exists():
        state.preproc = joblib.load(preproc_path)
        print(f"[ml] Preproc chargé: {preproc_path}")
    else:
        print(f"[ml] Preproc introuvable: {preproc_path}")

    if rank_path.exists():
        state.rank_model = joblib.load(rank_path)
        state.rank_model_path = str(rank_path)
        print(f"[ml] Rank model chargé: {rank_path}")
    else:
        print(f"[ml] Rank model introuvable: {rank_path}")
    return state

def load_df():
    df_etab = db.extract_table(db.engine,"etab")
    df_options =db.extract_table(db.engine,"options")
    df_embed = db.extract_table(db.engine,"etab_embedding")
    df_horaire = db.extract_table(db.engine,"opening_period")

    etab_features = df_etab[['id_etab', 'rating', 'priceLevel', 'latitude', 'longitude', 'adresse',
                             'editorialSummary_text', 'start_price']].copy()

    price_mapping = {
        'PRICE_LEVEL_INEXPENSIVE': 1, 'PRICE_LEVEL_MODERATE': 2,
        'PRICE_LEVEL_EXPENSIVE': 3, 'PRICE_LEVEL_VERY_EXPENSIVE': 4
    }
    etab_features['priceLevel'] = etab_features['priceLevel'].map(price_mapping)

    etab_features['priceLevel'] = etab_features.apply(fx.determine_price_level_row, axis=1)

    distribution = etab_features['priceLevel'].value_counts(normalize=True)
    if not distribution.empty and etab_features['priceLevel'].isnull().any():
        niveaux_existants = distribution.index
        probabilites = distribution.values
        nb_nan = etab_features['priceLevel'].isnull().sum()
        valeurs = np.random.choice(a=niveaux_existants, size=nb_nan, p=probabilites)
        etab_features.loc[etab_features['priceLevel'].isnull(), 'priceLevel'] = valeurs

    etab_features['code_postal'] = etab_features['adresse'].str.extract(r'(\b\d{5}\b)', expand=False)
    if etab_features['code_postal'].isnull().any():
        mode_cp = etab_features['code_postal'].mode()
        if not mode_cp.empty:
            etab_features['code_postal'].fillna(mode_cp.iloc[0], inplace=True)

    etab_features.drop(columns=['adresse'], inplace=True)
    etab_features['rating'].fillna(etab_features['rating'].mean(), inplace=True)
    etab_features['start_price'].fillna(0, inplace=True)

    # --- options_features (bool) ---
    bool_cols = [
        'allowsDogs','delivery','goodForChildren','goodForGroups','goodForWatchingSports',
        'outdoorSeating','reservable','restroom','servesVegetarianFood','servesBrunch',
        'servesBreakfast','servesDinner','servesLunch'
    ]
    options_features = df_options[['id_etab'] + [c for c in bool_cols if c in df_options.columns]].copy()
    for c in [col for col in bool_cols if col in options_features.columns]:
        options_features[c].fillna(False, inplace=True)
    if 'restroom' in options_features.columns:
        options_features['restroom'].fillna(True, inplace=True)

    # --- horaires -> profils 'ouvert_*' ---
    horaire_features = fx.compute_opening_profile(df_etab[['id_etab']].copy(), df_horaire)

    for d in (etab_features, options_features, df_embed, horaire_features):
        if 'id_etab' in d.columns:
            d['id_etab'] = pd.to_numeric(d['id_etab'], errors='coerce')

    # Merge
    df_final = etab_features.merge(options_features, on="id_etab", how="left")
    df_final = df_final.merge(horaire_features, on="id_etab", how="left")

    # --- merge embeddings ---
    df_final = df_final.merge(df_embed, on="id_etab", how="left")

    for c in [c for c in bool_cols if c in df_final.columns]:
        df_final[c] = df_final[c].fillna(False).astype(bool)

    if "desc_embed" in df_final.columns:
        df_final["desc_embed"] = df_final["desc_embed"].apply(utils.to_np1d)
    if "rev_embeds" in df_final.columns:
        df_final["rev_embeds"] = df_final["rev_embeds"].apply(utils.to_list_np)

    df_final = fx.sanitize_df_columns(df_final)
    if 'id_etab' in df_final.columns:
        df_final['id_etab'] = pd.to_numeric(df_final['id_etab'], errors='coerce')
        df_final = df_final.sort_values('id_etab').reset_index(drop=True)
    return df_final

def _log_table_df(df: pd.DataFrame, path: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    mlflow.log_text(buf.getvalue(), path)

def _ensure_mlflow():
    global _MLFLOW_READY, _MLFLOW_EXP_ID
    if _MLFLOW_READY and _MLFLOW_EXP_ID:
        return True

    uri = os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:5000"
    exp_name = os.getenv("MLFLOW_EXPERIMENT", "restaurant-api")
    try:
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = client.create_experiment(exp_name)
        else:
            exp_id = exp.experiment_id

        # Très important : fixe l'expérience par défaut du process
        mlflow.set_experiment(exp_name)

        _MLFLOW_EXP_ID = exp_id
        _MLFLOW_READY = True
        return True
    except Exception:
        return False
    
def log_prediction_event(prediction, form_dict, scores, used_ml: bool, latency_ms: int, model_version: str|None):
    """
    Log d'une requête d'inférence (une prédiction).
    - prediction: instance schema.Prediction (avec .items)
    - form_dict:  dict déjà nettoyé (pas d'info perso !)
    - scores:     np.array des scores, aligné au DF catalogue
    """
    tags = {
        "stage": "inference",
        "endpoint": "/predict",
        "prediction_id": str(prediction.id) if prediction.id else "na",
        "model_version": model_version or "dev",
        "used_ml": str(bool(used_ml)),
    }
    params = {
        "k": int(prediction.k),
    }
    metrics = {
         "latency_ms": float(latency_ms),
    }
    if not _ensure_mlflow():
        return
    with mlflow.start_run(
        run_name=f"predict:{tags['prediction_id']}",
        experiment_id=_MLFLOW_EXP_ID,          # <-- ici
        nested=True,
        tags=tags
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(form_dict, "inputs/form.json")
        topk_df = pd.DataFrame([{"rank": it.rank, "etab_id": it.etab_id, "score": it.score}
                                for it in prediction.items])
        _log_table_df(topk_df, "outputs/topk.csv")
        return run.info.run_id

def log_feedback_rating(prediction_id,rating,k= None,model_version= None,user_id= None,
    comment=None,use_active_run_if_any= True,):
    """
    Log du feedback global (0..5) dans MLflow.
    - Si un run est déjà actif (ex: ouvert pendant /predict) on log dedans.
    - Sinon on crée un run *court* dédié au feedback (nested si possible).

    Ne casse jamais l’API si MLflow n’est pas configuré.
    """

    rating = max(0, min(5, int(rating)))
    tags = {
        "stage": "feedback",
        "endpoint": "/feedback",
        "prediction_id": str(prediction_id),
        "model_version": model_version or "dev",
        "user_id": str(user_id) if user_id is not None else "na",
    }
    params = {"k_at_predict": int(k) if k is not None else None}
    metrics = {
        "user_rating": float(rating),
        "user_rating_norm": float(rating) / 5.0,
        "user_satisfied": 1.0 if rating >= 4 else 0.0,
    }
    if not _ensure_mlflow():
        return

    if comment:
        tags["feedback_comment_len"] = str(len(comment))

    active_run = mlflow.active_run()
    if use_active_run_if_any and active_run is not None:
        mlflow.set_tags({k: v for k, v in tags.items() if v is not None})
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics(metrics)
        return

    with mlflow.start_run(
        run_name=f"feedback:{prediction_id}",
        experiment_id=_MLFLOW_EXP_ID,         # <-- ici
        nested=True
    ):
        mlflow.set_tags({k: v for k, v in tags.items() if v is not None})
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics(metrics)

_PRICE_MAP_STR_TO_INT = {
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}
_DAY_FR = {
    1: "Lundi",
    2: "Mardi",
    3: "Mercredi",
    4: "Jeudi",
    5: "Vendredi",
    6: "Samedi",
    0: "Dimanche",   # NOTE: cohérent avec ton mapping existant {0: dimanche, 1: lundi, ...}
}
_DAY_ORDER = [1, 2, 3, 4, 5, 6, 0]  # Lundi → Dimanche

_OPTION_BOOL_COLS = (
    "allowsDogs", "delivery", "goodForChildren", "goodForGroups",
    "goodForWatchingSports", "outdoorSeating", "reservable", "restroom",
    "servesVegetarianFood", "servesBrunch", "servesBreakfast",
    "servesDinner", "servesLunch",
)

def _price_to_int_and_symbol(v: Optional[str | int | float]):
    """
    Convertit priceLevel en (niveau_int, symbole €), si possible.
    Accepte déjà un entier/float ou une constante texte PRICE_LEVEL_*.
    """
    if v is None:
        return None, None
    if isinstance(v, (int, float)):
        lvl = int(v)
        if 1 <= lvl <= 4:
            return lvl, "€" * lvl
        return lvl, None
    s = str(v).strip()
    if s in _PRICE_MAP_STR_TO_INT:
        lvl = _PRICE_MAP_STR_TO_INT[s]
        return lvl, "€" * lvl
    # parfois stocké comme "$", "$$", ...
    if set(s) == {"$"}:
        lvl = len(s)
        return lvl, "€" * lvl
    # fallback
    try:
        lvl = int(s)
        return lvl, ("€" * lvl) if 1 <= lvl <= 4 else None
    except ValueError:
        return None, None

def _fmt_time(hour: Optional[int], minute: Optional[int]) -> Optional[str]:
    if hour is None or minute is None:
        return None
    if minute == 0:
        return f"{hour}h"
    return f"{hour}h{minute:02d}"

def _build_horaires(periods: List[models.OpeningPeriod]) -> List[str]:
    """
    Construit des lignes 'Jour 10h–19h, 20h–22h'…
    Ignore les périodes qui traversent le jour (open_day != close_day) pour rester simple,
    comme dans ton code précédent.
    """
    by_day: Dict[int, List[str]] = {k: [] for k in _DAY_ORDER}
    for p in periods or []:
        if p.open_day is None or p.close_day is None:
            continue
        if p.open_day != p.close_day:
            # simplification (même logique que ton code de profil): on ignore les spans multi-jours
            continue
        t1 = _fmt_time(p.open_hour, p.open_minute)
        t2 = _fmt_time(p.close_hour, p.close_minute)
        if t1 and t2:
            by_day[p.open_day].append(f"{t1}–{t2}")

    lines = []
    for d in _DAY_ORDER:
        nom = _DAY_FR.get(d, f"Jour{d}")
        if by_day[d]:
            lines.append(f"{nom}\t{', '.join(by_day[d])}")
        else:
            lines.append(f"{nom}\tFermé")
    return lines

def _options_true_list(opt: Optional[models.Options]) -> List[str]:
    if not opt:
        return []
    out = []
    for c in _OPTION_BOOL_COLS:
        if getattr(opt, c, None) is True:
            out.append(c)
    return out

def get_etablissement_details(db: Session, id_etab: int) -> Dict[str, Any]:
    """
    Retourne un JSON avec:
      - nom, adresse, description/editorialSummary_text
      - rating, priceLevel (int + symbole), start_price
      - téléphone, site, géoloc
      - options_actives: noms des colonnes options à True
      - horaires: liste de 7 lignes "Jour\t(plages ou Fermé)" en FR
    """
    etab = (
        db.query(models.Etablissement)
        .options(
            selectinload(models.Etablissement.options),
            selectinload(models.Etablissement.opening_periods),
        )
        .filter(models.Etablissement.id_etab == id_etab)
        .first()
    )
    if not etab:
        return {"error": "etablissement introuvable", "id_etab": id_etab}

    lvl_int, lvl_sym = _price_to_int_and_symbol(etab.priceLevel)

    desc = etab.editorialSummary_text or etab.description

    payload: Dict[str, Any] = {
        "id_etab": etab.id_etab,
        "nom": etab.nom,
        "adresse": etab.adresse,
        "telephone": etab.internationalPhoneNumber,
        "site_web": etab.websiteUri,
        "description": desc,
        "rating": etab.rating,
        "priceLevel": lvl_int,         # niveau numérique si possible (1..4)
        "priceLevel_symbole": lvl_sym, # "€", "€€", ...
        "startPrice": etab.start_price,
        "endPrice": etab.end_price,
        "geo": {"lat": etab.latitude, "lng": etab.longitude},
        "options_actives": _options_true_list(etab.options),
        "horaires": _build_horaires(etab.opening_periods or []),
    }
    return payload

def get_etablissements_details_bulk(db: Session, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}

    etablissements = db.query(models.Etablissement).filter(models.Etablissement.id_etab.in_(ids)).all()
    if not etablissements:
        return {}

    etab_map = {etab.id_etab: etab for etab in etablissements}
    
    options_map = {
        opt.id_etab: opt for opt in 
        db.query(models.Options).filter(models.Options.id_etab.in_(ids)).all()
    }

    horaires_list = db.query(models.OpeningPeriod).filter(models.OpeningPeriod.id_etab.in_(ids)).all()
    horaires_map = {}
    for h in horaires_list:
        if h.id_etab not in horaires_map:
            horaires_map[h.id_etab] = []
        horaires_map[h.id_etab].append(h)

    out = {}
    for etab_id, etab in etab_map.items():
        lvl_int, lvl_sym = _price_to_int_and_symbol(etab.priceLevel)
        desc = etab.editorialSummary_text or etab.description
        
        etab_options = options_map.get(etab_id)
        etab_horaires = horaires_map.get(etab_id, [])

        out[etab.id_etab] = {
            "id_etab": etab.id_etab,
            "nom": etab.nom,
            "adresse": etab.adresse,
            "telephone": etab.internationalPhoneNumber,
            "site_web": etab.websiteUri,
            "description": desc,
            "rating": etab.rating,
            "priceLevel": lvl_int,
            "priceLevel_symbole": lvl_sym,
            "startPrice": etab.start_price,
            "endPrice": etab.end_price,
            "geo": {"lat": etab.latitude, "lng": etab.longitude},
            "options_actives": _options_true_list(etab_options),
            "horaires": _build_horaires(etab_horaires),
        }
    return out

def ensure_etabs_exist(db: Session, ids: list[int]):
    """
    S'assure que chaque id de 'ids' existe dans la table Etab.
    Crée des lignes minimales si besoin.
    Adapte les champs si ton modèle a des NOT NULL obligatoires.
    """
    if not ids:
        return
    ids = [int(x) for x in ids]
    existing = {
        x for (x,) in db.query(models.Etablissement.id_etab)
                        .filter(models.Etablissement.id_etab.in_(ids))
                        .all()
    }
    to_add = [i for i in ids if i not in existing]
    if to_add:
        # Si ton modèle a des colonnes NOT NULL, mets des valeurs par défaut ici.
        new_rows = [models.Etablissement(id_etab=i) for i in to_add]
        db.bulk_save_objects(new_rows)
        db.flush()