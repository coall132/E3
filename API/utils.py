from typing import Any, List
import json
import numpy as np
import pandas as pd

def _parse_vec(v: Any):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=float)
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return np.asarray(json.loads(v), dtype=float)
        except Exception:
            return None
    return None

def _parse_mat(m: Any):
    if m is None:
        return []
    if isinstance(m, (bytes, bytearray)):
        m = m.decode("utf-8", errors="ignore")
    if isinstance(m, str):
        m = m.strip()
        if not m:
            return []
        try:
            m = json.loads(m)
        except Exception:
            return []
    if isinstance(m, (list, tuple)):
        out = []
        for row in m:
            try:
                out.append(np.asarray(row, dtype=float))
            except Exception:
                pass
        return out
    return []

def determine_price_level(row):
    if pd.notna(row['start_price']):
        price = row['start_price']
        if price < 15:
            return 1
        elif price > 15:
            return 2
        elif price > 20:
            return 3
    else :
        return np.nan
    
def calculer_profil_ouverture(row, df_horaires, jours, creneaux):
    etab_id = row['id_etab']  
    profil = {'id_etab': etab_id}
    
    for j in jours.values():
        for c in creneaux.keys():
            profil[f"ouvert_{j}_{c}"] = 0
    
    horaires_etab = df_horaires[df_horaires['id_etab'] == etab_id]
    
    if horaires_etab.empty:
        return pd.Series(profil)

    for _, periode in horaires_etab.iterrows():
        if periode['open_day'] != periode['close_day']:
            continue
        jour_nom = jours.get(periode['open_day'])
        if not jour_nom:
            continue
        for nom_creneau, (debut, fin) in creneaux.items():
            if periode['open_hour'] < fin and periode['close_hour'] > debut:
                profil[f"ouvert_{jour_nom}_{nom_creneau}"] = 1
                
    return pd.Series(profil)

def _predict_scores(model, X: np.ndarray) -> np.ndarray:
    try:
        s = model.decision_function(X)
        return np.asarray(s, float).ravel()
    except Exception:
        pass
    try:
        proba = model.predict_proba(X)
        proba = np.asarray(proba, float)
        return (proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.max(axis=1))
    except Exception:
        pass
    pred = model.predict(X)
    return np.asarray(pred, float).ravel()

def _align_df_to_cols(X_df: pd.DataFrame, feature_cols: List[str]):
    for c in feature_cols:
        if c not in X_df.columns:
            X_df[c] = 0.0
    return X_df[feature_cols]

def to_np1d(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            return np.asarray(json.loads(x), dtype=np.float32)
        except Exception:
            return None
    return None

def to_list_np(x):
    if isinstance(x, list):
        out = []
        for e in x:
            if isinstance(e, np.ndarray):
                out.append(e.astype(np.float32))
            elif isinstance(e, list):
                out.append(np.asarray(e, dtype=np.float32))
            elif isinstance(e, str) and e.strip().startswith("["):
                try:
                    out.append(np.asarray(json.loads(e), dtype=np.float32))
                except Exception:
                    pass
        return out if out else None
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            L = json.loads(x)
            return [np.asarray(e, dtype=np.float32) for e in L]
        except Exception:
            return None
    return None


class _StubSentModel:
    def __init__(self, dim: int = 1024):
        self.dim = int(dim)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        # vecteurs nuls de la bonne dimension
        return [np.zeros(self.dim, dtype=np.float32) for _ in texts]

def _infer_embed_dim(df: pd.DataFrame, default: int = 1024) -> int:
    # essaie desc_embed, sinon ancres si tu en as, sinon fallback
    if "desc_embed" in df.columns:
        for v in df["desc_embed"]:
            if isinstance(v, np.ndarray) and v.ndim == 1:
                return int(v.shape[0])