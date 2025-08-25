from sqlalchemy.orm import Session, joinedload, selectinload
import secrets, base64, time
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import os
from fastapi import FastAPI, Depends, HTTPException, Security, status, Query
from argon2 import PasswordHasher, exceptions as argon_exc
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow

try:
    from . import models
    from . import database
    from . import schema
    from . import benchmark_3 as bm
    from . import utils
    from .main import app
except:
    import models
    import database as db
    import schema
    import benchmark_3 as bm
    import utils
    from main import app

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)  # pour /auth/token uniquement
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
ph = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=8)

API_STATIC_KEY = os.getenv("API_STATIC_KEY", "coall")  # pour échanger contre un token
JWT_SECRET = os.getenv("JWT_SECRET", "coall")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))

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
    state.preproc = getattr(bm, "preproc", None)
    state.preproc_factory = (
        getattr(bm, "build_preproc", None) or
        getattr(bm, "make_preproc", None)
        )
    state.sent_model = getattr(bm, "model", None)
    if not (state.preproc or state.preproc_factory):
            raise RuntimeError("Aucun préprocesseur trouvé dans benchmark_3 (preproc ou build_preproc).")
    path = os.getenv("RANK_MODEL_PATH", str(Path("artifacts") / "linear_svc_pointwise.joblib"))
    state.rank_model_path = path
    skip_rank = os.getenv("SKIP_RANK_MODEL", "0") == "1"
    if not skip_rank and Path(path).exists():
        try:
            state.rank_model = joblib.load(path)
            print(f"[ml] Rank model chargé: {path}")
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle: {path} ({e})") from e
    else:
        print(f"[ml] Rank model ignoré (fichier absent ou SKIP_RANK_MODEL=1): {path}")
    return state

def load_df():
    df_etab = db.extract_table(db.engine,"etab")
    df_options =db.extract_table(db.engine,"options")
    df_embed = db.extract_table(db.engine,"etab_embedding")
    df_horaire = db.extract_table(db.engine,"opening_period")

    etab_features = df_etab[['id_etab', 'rating', 'priceLevel', 'latitude', 'longitude','adresse',"editorialSummary_text","start_price"]].copy()
    price_mapping = {
        'PRICE_LEVEL_INEXPENSIVE': 1, 'PRICE_LEVEL_MODERATE': 2,
        'PRICE_LEVEL_EXPENSIVE': 3, 'PRICE_LEVEL_VERY_EXPENSIVE': 4
    }
    etab_features['priceLevel'] = etab_features['priceLevel'].map(price_mapping)
    etab_features['priceLevel'] = etab_features.apply(utils.determine_price_level,axis=1)

    distribution = etab_features['priceLevel'].value_counts(normalize=True)
    niveaux_existants = distribution.index
    probabilites = distribution.values
    nb_nan_a_remplacer = etab_features['priceLevel'].isnull().sum()
    valeurs_aleatoires = np.random.choice(a=niveaux_existants,size=nb_nan_a_remplacer,p=probabilites)
    etab_features.loc[etab_features['priceLevel'].isnull(), 'priceLevel'] = valeurs_aleatoires

    etab_features['code_postal'] = etab_features['adresse'].str.extract(r'(\b\d{5}\b)', expand=False)
    etab_features['code_postal'].fillna(etab_features['code_postal'].mode()[0], inplace=True)
    etab_features.drop("adresse",axis=1,inplace=True)
    etab_features['rating'].fillna(etab_features['rating'].mean(), inplace=True)
    etab_features['start_price'].fillna(0, inplace=True)

    options_features = df_options.copy()

    options_features['allowsDogs'].fillna(False, inplace=True)
    options_features['delivery'].fillna(False, inplace=True)
    options_features['goodForChildren'].fillna(False, inplace=True)
    options_features['goodForGroups'].fillna(False, inplace=True)
    options_features['goodForWatchingSports'].fillna(False, inplace=True)
    options_features['outdoorSeating'].fillna(False, inplace=True)
    options_features['reservable'].fillna(False, inplace=True)
    options_features['restroom'].fillna(True, inplace=True)
    options_features['servesVegetarianFood'].fillna(False, inplace=True)
    options_features['servesBrunch'].fillna(False, inplace=True)
    options_features['servesBreakfast'].fillna(False, inplace=True)
    options_features['servesDinner'].fillna(False, inplace=True)
    options_features['servesLunch'].fillna(False, inplace=True)
    horaire_features=df_horaire.copy()

    horaire_features.dropna(subset=['close_hour', 'close_day'], inplace=True)
    for col in ['open_day', 'open_hour', 'close_day', 'close_hour']:
        horaire_features[col] = horaire_features[col].astype(int)

    jours = {0: "dimanche", 1: "lundi", 2: "mardi", 3: "mercredi", 4: "jeudi", 5: "vendredi", 6: "samedi"}
    creneaux = {"matin": (8, 11), "midi": (11, 14), "apres_midi": (14, 19), "soir": (19, 23)}

    horaire_features = df_etab.apply(utils.calculer_profil_ouverture,axis=1,df_horaires=horaire_features,jours=jours,creneaux=creneaux)

    df_final=pd.merge(etab_features,options_features,on="id_etab",how='left')
    df_final=pd.merge(df_final,horaire_features,on="id_etab",how='left')
    df_final_embed=pd.merge(df_final,df_embed,on="id_etab",how='left')
    if "desc_embed" in df_final_embed.columns:
        df_final_embed["desc_embed"] = df_final_embed["desc_embed"].apply(utils.to_np1d)
    if "rev_embeds" in df_final_embed.columns:
        df_final_embed["rev_embeds"] = df_final_embed["rev_embeds"].apply(utils.to_list_np)
    return df_final_embed

def log_prediction_mlflow(form, scores, sel_indices, df, used_ml, latency_ms, k):
    if not getattr(app.state, "MLFLOW_RUN", None): 
        return

    topk = []
    for r, i in enumerate(sel_indices, start=1):
        etab_id = int(df.iloc[i]["id_etab"]) if "id_etab" in df.columns else int(i)
        topk.append({"rank": r, "etab_id": etab_id, "score": float(scores[i])})

    mlflow.log_metrics({
        "latency_ms": latency_ms,
        "k": k,
        "used_ml": int(bool(used_ml)),
        "score_mean": float(np.mean(scores[sel_indices])),
        "score_std": float(np.std(scores[sel_indices])),
    })

    if "code_postal" in df.columns:
        div = len(set(df.iloc[sel_indices]["code_postal"].astype(str)))
        mlflow.log_metric("diversity_cp", int(div))

    artifact = {
        "form": {k: v for k, v in form.items() if k in ("price_level","code_postal","options","open","description")},
        "topk": topk
    }
    mlflow.log_dict(artifact, f"predictions/{int(time.time()*1000)}.json")