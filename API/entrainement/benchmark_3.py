#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import sys


from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import matplotlib.pyplot as plt

import API.entrainement.Extract
import API.utils 
from entrainement.fallback import FallbackRanker

FAST = os.getenv("E3_FAST_TEST", "0") == "1"
n_estimators = 20 if FAST else 200 
   
PROXY_W_REV = 0.5
PROXY_K = 2

EVAL_W_REV = 0.8
EVAL_K = 5
# =========================================================
#            UTILITAIRES GÉNÉRAUX (FORM / FEATURES)
# =========================================================
def _artifacts_dir():
    base = os.getenv("ARTIFACTS_DIR")
    d = Path(base) if base else (Path.cwd() / "artifacts")
    d.mkdir(parents=True, exist_ok=True)
    return d

def fget(form, key, default=None):
    if isinstance(form, dict):
        return form.get(key, default)
    if isinstance(form, pd.Series):
        return form.get(key, default)
    return getattr(form, key, default)

def sanitize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # force tout en str
    df.columns = [str(c) for c in df.columns]
    # enlève d’éventuels guillemets autour des noms
    df.rename(columns=lambda c: re.sub(r'^"+|"+$', '', c), inplace=True)
    return df

def _iter_forms(forms):
    if isinstance(forms, pd.DataFrame):
        for row in forms.itertuples(index=False):
            yield row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())
    elif isinstance(forms, list):
        for f in forms:
            yield f
    else:
        yield forms

def _first_code_postal(pcs):
    if isinstance(pcs, pd.Series):
        pcs = pcs.dropna().tolist()
    if isinstance(pcs, np.ndarray):
        pcs = pcs.tolist()
    if isinstance(pcs, (list, tuple, set)):
        for v in pcs:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ''
    if pcs is None:
        return ''
    return str(pcs).strip()

# =========================================================
#        PRÉPARATION DES DONNÉES (EX-2e FICHIER)
# =========================================================
def _jsonify_best(d: dict, drop_keys=("model",)) -> dict:
    """Renvoie un dict JSON-safe (sans objets sklearn, sans types NumPy)."""
    out = {}
    for k, v in d.items():
        if k in drop_keys:
            continue
        # numpy scalars -> python
        if isinstance(v, (np.generic,)):
            v = v.item()
        # numpy arrays -> listes
        if isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out

def determine_price_level_row(row):
    """
    Si 'start_price' est présent :
       < 15€ -> 1 ; 15-20€ -> 2 ; > 20€ -> 3
    Sinon : on garde la valeur (mapping) de priceLevel.
    """
    sp = row.get('start_price', np.nan)
    if pd.notna(sp):
        try:
            price = float(sp)
        except Exception:
            return np.nan
        if price < 15:
            return 1
        elif price <= 20:
            return 2
        else:
            return 3
    return row.get('priceLevel', np.nan)

def compute_opening_profile(df_etab: pd.DataFrame, df_horaires: pd.DataFrame) -> pd.DataFrame:
    jours = {0: "dimanche", 1: "lundi", 2: "mardi", 3: "mercredi", 4: "jeudi", 5: "vendredi", 6: "samedi"}
    creneaux = {"matin": (8, 11), "midi": (11, 14), "apres_midi": (14, 19), "soir": (19, 23)}

    hf = df_horaires.copy()
    # drop lignes incomplètes
    hf.dropna(subset=['close_hour', 'close_day'], inplace=True)
    for col in ['open_day', 'open_hour', 'close_day', 'close_hour']:
        hf[col] = hf[col].astype(int)

    def calculer_profil_ouverture(etab_row):
        etab_id = etab_row['id_etab']
        profil = {'id_etab': etab_id}
        for j in jours.values():
            for c in creneaux.keys():
                profil[f"ouvert_{j}_{c}"] = 0

        horaires_etab = hf[hf['id_etab'] == etab_id]
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

    return df_etab.apply(calculer_profil_ouverture, axis=1)

def load_and_prepare_catalog() -> pd.DataFrame:
    """
    Charge les tables via Extract.main() puis construit le df final prêt pour entrainement/éval :
      - etab_features (rating, priceLevel numérisé, code_postal, etc.)
      - options_features (booléens imputés)
      - horaire_features (profils 'ouvert_*')
      - merge embeddings (desc_embed, rev_embeds) -> conversion en np.ndarray
    """
    all_dfs = API.entrainement.Extract.main()
    df_etab      = all_dfs['etab'].copy()
    df_options   = all_dfs['options'].copy()
    df_embed     = all_dfs['etab_embedding'].copy()
    df_horaire   = all_dfs['opening_period'].copy()

    # --- etab_features ---
    etab_features = df_etab[['id_etab', 'rating', 'priceLevel', 'latitude', 'longitude', 'adresse',
                             'editorialSummary_text', 'start_price']].copy()

    price_mapping = {
        'PRICE_LEVEL_INEXPENSIVE': 1, 'PRICE_LEVEL_MODERATE': 2,
        'PRICE_LEVEL_EXPENSIVE': 3, 'PRICE_LEVEL_VERY_EXPENSIVE': 4
    }
    etab_features['priceLevel'] = etab_features['priceLevel'].map(price_mapping)

    etab_features['priceLevel'] = etab_features.apply(determine_price_level_row, axis=1)

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
    horaire_features = compute_opening_profile(df_etab[['id_etab']].copy(), df_horaire)

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

    # Normalise les embeddings en np.ndarray (desc_embed: vecteur, rev_embeds: liste de vecteurs)
    def _as_np_vec(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.array([], dtype=np.float32)
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            return np.array([], dtype=np.float32)
        if arr.ndim > 1:
            # si jamais un 2D tombe ici, on fait une moyenne
            return arr.mean(axis=0).astype(np.float32)
        return arr

    def _as_list_of_vecs(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return []
        if isinstance(x, list):
            out = []
            for v in x:
                vv = np.asarray(v, dtype=np.float32)
                if vv.ndim == 1 and vv.size > 0:
                    out.append(vv)
            return out
        # si c'est un seul vecteur, on le met dans une liste
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1 and arr.size > 0:
            return [arr]
        return []

    if 'desc_embed' in df_final.columns:
        df_final['desc_embed'] = df_final['desc_embed'].apply(_as_np_vec)
    if 'rev_embeds' in df_final.columns:
        df_final['rev_embeds'] = df_final['rev_embeds'].apply(_as_list_of_vecs)
    df_final = sanitize_df_columns(df_final)
    if 'id_etab' in df_final.columns:
        df_final['id_etab'] = pd.to_numeric(df_final['id_etab'], errors='coerce')
        df_final = df_final.sort_values('id_etab').reset_index(drop=True)
    return df_final

# =========================================================
#      BLOC TEXTE / EMBEDDINGS POUR LE SCORING (IA)
# =========================================================
SENT_MODEL = SentenceTransformer('BAAI/bge-m3')

def topk_mean_cosine(mat_or_list, z, k=3):
    if mat_or_list is None:
        return None
    if isinstance(mat_or_list, list):
        if len(mat_or_list) == 0:
            return None
        M = np.vstack([np.asarray(v, dtype=np.float32) for v in mat_or_list])
    else:
        M = np.asarray(mat_or_list, dtype=np.float32)
        if M.ndim == 1:
            M = M[None, :]
    if M.size == 0:
        return None
    sims = M @ z.astype(np.float32)
    k = min(int(k), len(sims))
    if k <= 0:
        return None
    idx = np.argpartition(sims, -k)[-k:]
    return float(np.mean(sims[idx]))

def score_text(df, form, model, w_rev=0.6, w_desc=0.4, k=3, missing_cos=0.0):
    q = (form.get("description") or "").strip()
    N = len(df)
    if not q:
        return np.ones(N, dtype=np.float32)

    z = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    # desc
    desc_list = df.get("desc_embed", None)
    cos_desc = np.full(N, missing_cos, dtype=np.float32)
    if desc_list is not None:
        for i, v in enumerate(desc_list):
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 0:
                cos_desc[i] = _cos01_safe(v, z)

    # rev top-k
    rev_list = df.get("rev_embeds", None)
    cos_rev = np.full(N, missing_cos, dtype=np.float32)
    if rev_list is not None:
        for i, L in enumerate(rev_list):
            if isinstance(L, list) and len(L) > 0:
                vals = [_cos01_safe(r, z) for r in L
                        if isinstance(r, np.ndarray) and r.ndim == 1 and r.size > 0]
                if vals:
                    k_ = min(k, len(vals))
                    cos_rev[i] = float(np.sort(np.asarray(vals, np.float32))[-k_:].mean())

    return (w_desc * cos_desc + w_rev * cos_rev).astype(np.float32)

def h_price_vector_simple(df, form):
    lvl_f = fget(form, 'price_level', None)
    if lvl_f is None:
        return np.full(len(df), 0.5, dtype=float)
    diff = (df['priceLevel'].astype(float) - float(lvl_f)).abs()
    return (1.0 - (diff/3.0)).clip(0.0, 1.0).to_numpy(dtype=float)

def h_rating_vector(df, alpha=20.0):
    r = df['rating'].astype(float).fillna(2.5) if 'rating' in df.columns else pd.Series(2.5, index=df.index)
    mu = float(r.mean()) if r.notna().any() else 0.0
    if 'rev_embeds' in df.columns:
        n = df['rev_embeds'].apply(lambda x: len(x) if isinstance(x, list) else 0).astype(float)
    else:
        n = pd.Series(1.0, index=df.index)
    r_star = (n*r + alpha*mu) / (n + alpha)
    return np.clip(r_star/5.0, 0.0, 1.0).to_numpy(dtype=float)

def h_city_vector(df, form):
    pcs = fget(form, 'code_postal', None)
    if pcs is None or 'code_postal' not in df.columns:
        return np.ones(len(df), dtype=float)
    if not isinstance(pcs, (list, tuple, set)):
        pcs = [pcs]
    pcs = [str(x).strip() for x in pcs if str(x).strip()]
    if not pcs:
        return np.full(len(df), 0.5, dtype=float)
    s = df['code_postal'].astype(str).str.strip()
    return s.isin(pcs).astype(float).to_numpy()

def _extract_requested_options(form, df_catalog):
    opts = fget(form, 'options', None)
    if isinstance(opts, str) and opts.strip():
        toks = re.split(r'[;,]\s*', opts.strip())
        return [t for t in toks if t in df_catalog.columns]
    if isinstance(opts, (list, tuple, set)):
        return [o for o in opts if o in df_catalog.columns]
    out = []
    for c in df_catalog.columns:
        if df_catalog[c].dtype == bool:
            v = fget(form, c, None)
            if isinstance(v, (bool, np.bool_)) and v:
                out.append(c)
            elif isinstance(v, str) and v.lower() in ('1','true','vrai','yes','oui'):
                out.append(c)
    return out

def h_opts_vector(df, form):
    req = _extract_requested_options(form, df)
    if not req:
        return np.full(len(df), 0.5, dtype=float)   
    sub = df[req].fillna(False).astype(int).to_numpy()
    return sub.mean(axis=1).astype(float)

def h_open_vector(df, form, unknown_value=0.5):
    col = fget(form, 'open', None)
    if not col or col not in df.columns:
        return np.full(len(df), 0.5, dtype=float)
    cols = [c for c in df.columns if c.startswith(col)]
    if not cols:
        return np.ones(len(df), dtype=float)
    M = df[cols].fillna(0).astype(int)
    has_any_open_info = (M.sum(axis=1) > 0).to_numpy()
    v = df[col].fillna(0).astype(int).to_numpy().astype(float)
    return np.where(has_any_open_info, v, float(unknown_value))

def score_func(df, form, model):
    return {
        "price":   h_price_vector_simple(df, form),
        "rating":  h_rating_vector(df, alpha=20.0),
        "city":    h_city_vector(df, form),
        "options": h_opts_vector(df, form),
        "open":    h_open_vector(df, form, unknown_value=0.5),
        "text":    score_text(df, form, model, w_rev=0.5, w_desc=0.5, k=3, missing_cos=0.0),
    }

def aggregate_gains(H: Dict[str, np.ndarray], weights: Dict[str, float]):
    keys = list(H)
    W = np.array([float(weights[k]) for k in keys], dtype=float)
    gains = np.zeros_like(np.asarray(H[keys[0]], dtype=float), dtype=float)
    for k, w in zip(keys, W):
        gains += w * np.asarray(H[k], dtype=float)
    return gains / W.sum()

def text_features(df, form, model, k=PROXY_K):
    q = (fget(form,'description','') or '').strip()
    N = len(df)
    if not q:
        return np.zeros((N, 2), dtype=np.float32)
    z = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    desc = df.get('desc_embed', [None]*N)
    cos_desc = np.array([float(v @ z) if isinstance(v, np.ndarray) and v.ndim==1 and v.size>0 else 0.0
                         for v in desc], dtype=np.float32)

    revs = df.get('rev_embeds', [None]*N)
    cos_rev = np.array([(topk_mean_cosine(r, z, k=k) or 0.0) for r in revs], dtype=np.float32)

    # Optionnel: map [-1,1] -> [0,1]
    cos_desc = (cos_desc + 1.0)/2.0
    cos_rev  = (cos_rev  + 1.0)/2.0
    return np.stack([cos_desc, cos_rev], axis=1)

def cos01_safe(v: np.ndarray, z: np.ndarray) -> float:
    """Cosinus robuste ∈ [0,1] avec alignement des dimensions et normalisation."""
    v = np.asarray(v, np.float32).ravel()
    z = np.asarray(z, np.float32).ravel()
    m = min(v.size, z.size)
    if m == 0:
        return 0.0
    vv, zz = v[:m], z[:m]
    nv = np.linalg.norm(vv); nz = np.linalg.norm(zz)
    if nv == 0.0 or nz == 0.0:
        return 0.0
    c = float(np.dot(vv / nv, zz / nz))          # c ∈ [-1,1]
    return 0.5 * (np.clip(c, -1.0, 1.0) + 1.0)   # → [0,1]

def text_features01(df, form, model, k=3, missing_cos01=0.0):
    """
    Retourne un array (N,2) dans [0,1] :
      [:,0] = cos_desc01,  [:,1] = cos_rev_topk01
    """
    q = (form.get('description') or '').strip()
    N = len(df)
    out = np.zeros((N, 2), dtype=np.float32)

    # pas de texte => neutre
    if not q:
        out[:] = 1.0
        return out

    z = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    # cos avec la description
    cos_d = np.full(N, np.nan, dtype=np.float32)
    desc_list = df.get('desc_embed', None)
    if desc_list is not None:
        for i, v in enumerate(desc_list):
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 0:
                cos_d[i] = cos01_safe(v, z)

    # cos top-k sur les reviews
    cos_r = np.full(N, np.nan, dtype=np.float32)
    rev_list = df.get('rev_embeds', None)
    if rev_list is not None:
        for i, L in enumerate(rev_list):
            if isinstance(L, list) and len(L) > 0:
                vals = [
                    cos01_safe(r, z)
                    for r in L
                    if isinstance(r, np.ndarray) and r.ndim == 1 and r.size > 0
                ]
                if vals:
                    k_ = min(k, len(vals))
                    cos_r[i] = float(np.sort(np.asarray(vals, np.float32))[-k_:].mean())

    # remplit les manquants (déjà en [0,1])
    cos_d = np.where(np.isnan(cos_d), missing_cos01, cos_d)
    cos_r = np.where(np.isnan(cos_r), missing_cos01, cos_r)
    out[:, 0] = cos_d
    out[:, 1] = cos_r
    return out

def pair_features(Zf, X_items, T, diff_scale=0.05):
    diff = np.abs(X_items - Zf.reshape(1, -1)) * float(diff_scale)  # (N, d)
    T = np.asarray(T, dtype=np.float32)                             # (N, 2) = [cos_desc01, cos_rev01]
    return np.hstack([diff, T]).astype(np.float32)

# =========================================================
#           PRÉPROC ITEM (colonnes classiques)
# =========================================================

def build_preproc_for_items(df: pd.DataFrame):
    BOOL_COLS = [
        'allowsDogs','delivery','goodForChildren','goodForGroups','goodForWatchingSports',
        'outdoorSeating','reservable','restroom','servesVegetarianFood','servesBrunch',
        'servesBreakfast','servesDinner','servesLunch'
    ]
    NUM_COLS = ['rating','start_price']
    lev = 'priceLevel'

    bool_cols_present = [c for c in BOOL_COLS if c in df.columns]
    bool_categories = [np.array([0, 1], dtype=int) for _ in bool_cols_present]

    # Numériques
    num_pipe  = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scale', StandardScaler())
    ])

    # Booléens -> OneHot (0/1)
    bool_pipe = Pipeline([
        ('toint', FunctionTransformer(API.utils.to_int_safe, validate=False, feature_names_out='one-to-one')),
        ('onehot', OneHotEncoder(
            categories=bool_categories,
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=True
        ))
    ])

    # priceLevel
    lev_pipe  = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scale', StandardScaler())
    ])

    # 🔹 Nouveau : code_postal (catégorie texte) -> OneHot
    #   - imputation au mode
    #   - handle_unknown='ignore' pour éviter les erreurs si un CP apparaît en prod
    cp_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    transformers = [
        ("num",  num_pipe,  [c for c in NUM_COLS if c in df.columns]),
        ("bool", bool_pipe, [c for c in BOOL_COLS if c in df.columns]),
        ("lev",  lev_pipe,  [lev] if lev in df.columns else []),
    ]

    # On n’ajoute cp que s’il existe dans le catalogue
    if 'code_postal' in df.columns:
        transformers.append(("cp", cp_pipe, ['code_postal']))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

# =========================================================
#          GÉNÉRATION DATASETS & MÉTRIQUES RANKING
# =========================================================
W_eval     = {'price':0.08,'rating':0.10,'options':0.10,'text':0.52,'city':0.15,'open':0.05}
W_proxy     = {'price':0.22,'rating':0.18,'options':0.10,'text':0.20,'city':0.10,'open':0.20}
tau = float(os.getenv("E3_TAU", "0.5"))

def form_to_row(form, df_catalog):
    row = {c: np.nan for c in df_catalog.columns}
    if 'type' in df_catalog.columns:
        row['type'] = fget(form, 'type', '')
    if 'priceLevel' in df_catalog.columns:
        lvl = fget(form, 'price_level', None)
        row['priceLevel'] = np.nan if lvl is None else float(lvl)
    if 'code_postal' in df_catalog.columns:
        row['code_postal'] = _first_code_postal(fget(form, 'code_postal', None))
    
    bool_cols = [c for c in df_catalog.columns if df_catalog[c].dtype == bool]
    for c in bool_cols:
        row[c] = False

    opts = fget(form, 'options', [])
    if isinstance(opts, str) and opts.strip():
        opts = [x.strip() for x in re.split(r'[;,]', opts)]
    if isinstance(opts, (list, tuple, set)):
        for c in opts:
            if c in df_catalog.columns:
                row[c] = True
    
    for c in ['rating','start_price']:
        if c in df_catalog.columns:
            row[c] = 0.0
    return pd.DataFrame([row])[df_catalog.columns]

def build_pointwise(forms_df, preproc, df, X_items, SENT_MODEL, tau=tau, W=W_proxy):
    X_list, y_list, sw_list, qid_list = [], [], [], []
    n_items = len(df)

    for qid, form in enumerate(_iter_forms(forms_df), start=0):
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

        T = text_features01(df, form, SENT_MODEL, k=PROXY_K)

        H_no_text = {
            'price'  : h_price_vector_simple(df, form),
            'rating' : h_rating_vector(df),
            'city'   : h_city_vector(df, form),
            'options': h_opts_vector(df, form),
            'open'   : h_open_vector(df, form, unknown_value=1.0), 
        }
        text_proxy = PROXY_W_REV * T[:, 1] + (1.0 - PROXY_W_REV) * T[:, 0]
        gains = aggregate_gains({**H_no_text, 'text': text_proxy}, W_proxy)

        y  = (gains >= tau).astype(np.int32)                
        sw = (np.abs(gains - tau) + 1e-3).astype(np.float32)

        # Features modèle
        Xq = pair_features(Zf, X_items, T)

        X_list.append(Xq); y_list.append(y); sw_list.append(sw)
        qid_list.append(np.full(n_items, qid, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    sw = np.concatenate(sw_list)
    qid = np.concatenate(qid_list)
    return X, y, sw, qid

def build_pairwise(forms_df, preproc, df, X_items, SENT_MODEL, tau=tau, W=W_proxy, top_m=10, bot_m=10):
    Xp, yp, wp = [], [], []

    for form in _iter_forms(forms_df):
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

        # Non-texte
        H_no_text = {
            'price':   h_price_vector_simple(df, form),
            'rating':  h_rating_vector(df),
            'city':    h_city_vector(df, form),
            'options': h_opts_vector(df, form),
            'open':    h_open_vector(df, form, unknown_value=0.5),
        }

        # Texte
        T = text_features01(df, form, SENT_MODEL, k=PROXY_K)
        text_proxy = PROXY_W_REV * T[:, 1] + (1.0 - PROXY_W_REV) * T[:, 0]

        # Gains pour choisir pos/neg
        H_lbl = {**H_no_text, 'text': text_proxy}
        gains = aggregate_gains(H_lbl, W)

        order = np.argsort(gains)
        n = len(order)
        pos_idx = order[::-1][:min(top_m, n)]
        neg_idx = order[:min(bot_m, n)]

        # Features modèle (par item)
        Xq = pair_features(Zf, X_items, T)

        # Construit les paires
        for ip in pos_idx:
            for ineg in neg_idx:
                if ip == ineg:
                    continue
                Xp.append(Xq[ip] - Xq[ineg])
                yp.append(1)
                # poids = |différence de gain|  (>= 0)
                w = float(abs(gains[ip] - gains[ineg]))
                wp.append(w)

        # Filet de sécurité : si rien n’a été ajouté (dataset trop jouet)
        if not Xp:
            best, worst = int(order[-1]), int(order[0])
            if best != worst:
                Xp.append(Xq[best] - Xq[worst])
                yp.append(1)
                w = float(abs(gains[best] - gains[worst]))
                wp.append(w if w > 0.0 else 1e-6)
            else:
                # extrême secours : paire nulle mais poids epsilon
                Xp.append(np.zeros_like(Xq[0]))
                yp.append(1)
                wp.append(1e-6)

    Xp = np.vstack(Xp).astype(np.float32)
    yp = np.ones(len(yp), dtype=int)
    wp = np.asarray(wp, dtype=float)

def gains_to_labels_per_query(gains: np.ndarray, q=(0.50, 0.75, 0.90)) -> np.ndarray:
    """
    Quantifie des gains continus en grades {0,1,2,3} par requête.
    q = quantiles -> 4 classes.
    Garantit au moins 2 grades présents.
    """
    gains = np.asarray(gains, dtype=float)
    # bornage soft
    if np.allclose(gains.min(), gains.max()):
        # tout égal -> force 1 positif pour éviter groupe dégénéré
        y = np.zeros_like(gains, dtype=int)
        y[np.argmax(gains)] = 1
        return y

    qs = np.quantile(gains, q)
    y = np.digitize(gains, bins=qs).astype(int)  # 0..len(q)
    if len(np.unique(y)) < 2:
        # force un peu de diversité
        top = np.argpartition(gains, -3)[-3:]
        y[top] = 1
    return y

def build_listwise(forms_df, preproc, df, X_items, SENT_MODEL, W=W_proxy, jitter=1e-6):
    X_list, y_list, groups = [], [], []
    n_items = len(df)

    for form in _iter_forms(forms_df):
        # ----- embedding du formulaire -----
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

        # ----- composantes non-texte (features "règles") -----
        H_no_text = {
            'price':   h_price_vector_simple(df, form),
            'rating':  h_rating_vector(df),
            'city':    h_city_vector(df, form),
            'options': h_opts_vector(df, form),
            'open':    h_open_vector(df, form, unknown_value=0.5),
        }

        # ----- TEXTE : features brutes pour le modèle -----
        # renvoie un tableau (N, 2) dans [0,1] : [cos_desc01, cos_rev01]
        T_feat = text_features01(df, form, SENT_MODEL, k=PROXY_K)
        cos_desc01 = T_feat[:, 0]
        cos_rev01  = T_feat[:, 1]

        # ----- TEXTE : agrégat UNIQUEMENT pour fabriquer les labels -----
        text_proxy = PROXY_W_REV * cos_rev01 + (1.0 - PROXY_W_REV) * cos_desc01
        H_lbl = {**H_no_text, 'text': text_proxy}

        # gains -> labels (listwise)
        gains_eval = aggregate_gains(H_lbl, W).astype(float)
        if jitter:
            gains_eval = gains_eval + float(jitter) * np.random.randn(len(gains_eval))
        yq = gains_to_labels_per_query(gains_eval, q=(0.50, 0.75, 0.90))

        # ----- FEATURES modèle : sans agrégat texte -----
        Xq = pair_features(Zf, X_items, T_feat)  # T_feat (N,2)

        X_list.append(Xq)
        y_list.append(yq)
        groups.append(n_items)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, groups


def precision_at_k(pred, gold, k):
    idx = np.asarray(pred, dtype=int)[:k]
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    hits = np.isin(idx, list(gold)).sum()
    return float(hits) / float(k) if k > 0 else 0.0

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return r[0] + (r[1:] / np.log2(np.arange(2, r.size + 1))).sum()

def ndcg_gain_at_k(pred, gains, k):
    idx = np.asarray(pred, dtype=int)[:k]
    rel = np.asarray(gains, dtype=float)
    dcg = dcg_at_k(rel[idx], k)
    ideal = np.sort(rel)[::-1][:k]
    idcg = dcg_at_k(ideal, k)
    return float(dcg / idcg) if idcg > 0 else 0.0

def ndcg_binary_at_k(pred, gold, k):
    n = len(pred) 
    idx = np.asarray(pred, dtype=int)[:k] 
    rel = np.zeros(n, dtype=float) 
    if not isinstance(gold, (set, frozenset)): 
        gold = set(gold) 
    if gold: 
        rel[list(gold)] = 1.0 
        dcg = dcg_at_k(rel[idx], k) 
        ideal_k = min(k, int(rel.sum())) 
    if ideal_k == 0: 
        return 0.0 
    idcg = dcg_at_k(np.ones(ideal_k, dtype=float), ideal_k) 
    return float(dcg / idcg) if idcg > 0 else 0.0

def recall_at_k(pred, gold, k):
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    if len(gold) == 0:
        return 0.0
    idx = np.asarray(pred, dtype=int)[:k]
    hits = np.isin(idx, list(gold)).sum()
    return float(hits) / float(len(gold))

def hitrate_at_k(pred, gold, k):
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    if len(gold) == 0:
        return 0.0
    idx = np.asarray(pred, dtype=int)[:k]
    return 1.0 if np.isin(idx, list(gold)).any() else 0.0

def ap_at_k(pred, gold, k):
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    if len(gold) == 0:
        return 0.0
    idx = np.asarray(pred, dtype=int)[:k]
    hits = 0
    precisions = []
    for i, it in enumerate(idx, start=1):
        if it in gold:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    denom = min(k, len(gold))
    return float(np.sum(precisions) / denom)

def mrr_at_k(pred, gold, k):
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    idx = np.asarray(pred, dtype=int)[:k]
    for i, it in enumerate(idx, start=1):
        if it in gold:
            return 1.0 / i
    return 0.0

def dcg_binary_at_k(pred, gold, k):
    """
    DCG binaire sur le top-k.
    gold : set d’indices pertinents
    """
    idx = np.asarray(pred, dtype=int)[:k]
    rel = np.array([1.0 if i in gold else 0.0 for i in idx], dtype=float)
    return dcg_at_k(rel, k)


def _count_inversions(arr):
    # décompte d’inversions O(n log n) (merge sort)
    def merge_count(a):
        n = len(a)
        if n <= 1:
            return a, 0
        m = n // 2
        left, inv_l = merge_count(a[:m])
        right, inv_r = merge_count(a[m:])
        merged = []
        i = j = inv = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
                inv += len(left) - i
        merged.extend(left[i:]); merged.extend(right[j:])
        return merged, inv_l + inv_r + inv
    _, inv = merge_count(list(arr))
    return inv

def _active_weights(form, base_w, df):
    w = base_w.copy()
    if fget(form, 'price_level', None) is None: w['price'] = 0.0
    if not _extract_requested_options(form, df):    w['options'] = 0.0
    if not fget(form, 'code_postal', None):         w['city'] = 0.0
    if not fget(form, 'open', None):                w['open'] = 0.0
    s = sum(w.values()); 
    if s > 0: w = {k: v/s for k, v in w.items()}
    return w

def kendall_tau_at_k(pred, gains_eval, k):
    """
    Kendall’s tau (tau-a) sur le top-k, en comparant l’ordre prédit au top-k idéal (tri par gains_eval).
    On calcule τ sur l’**intersection** des items (pour éviter les éléments hors top-k idéal).
    """
    pred = np.asarray(pred, dtype=int)[:k]
    true_topk = np.argsort(gains_eval)[::-1][:k]
    pos_true = {itm: r for r, itm in enumerate(true_topk)}
    common = [itm for itm in pred if itm in pos_true]
    n = len(common)
    if n < 2:
        return 0.0
    # séquence des rangs "vrais" dans l’ordre prédit
    seq = [pos_true[itm] for itm in common]
    inv = _count_inversions(seq)
    total = n * (n - 1) // 2
    return 1.0 - 2.0 * inv / total


def _bootstrap_ci(values, n_boot=300, alpha=0.05, rng=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (0.0, 0.0, 0.0)
    rng = rng or np.random.default_rng(123)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    lo = means[int((alpha/2)*n_boot)]
    hi = means[int((1-alpha/2)*n_boot)-1]
    return (float(values.mean()), float(lo), float(hi))

def eval_benchmark(forms_df, preproc, df, X_items, SENT_MODEL, models, k=5,
                   tau_q=0.85, use_top_m=None, jitter=1e-6, n_boot=300):
    model_names = ["Random", "RuleProxy"] + list(models.keys())
    per_model = {name: [] for name in model_names}

    for form in _iter_forms(forms_df):
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]
        H = {
            'price':  h_price_vector_simple(df, form),
            'rating': h_rating_vector(df),
            'city':   h_city_vector(df, form),
            'options':h_opts_vector(df, form),
            'open':   h_open_vector(df, form, unknown_value=0.5),
        }
        T_eval  = text_features01(df, form, SENT_MODEL, k=EVAL_K) 
        cos_desc01 = T_eval[:, 0]
        cos_rev01  = T_eval[:, 1]

        # Agrégat texte UNIQUEMENT pour le gold (métriques)
        text_eval = EVAL_W_REV * cos_rev01 + (1.0 - EVAL_W_REV) * cos_desc01
        H_eval = {**H, 'text': text_eval}

        W_eval_q  = _active_weights(form, W_eval,  df)
        gains_eval = aggregate_gains(H_eval, W_eval_q).astype(float)
        if jitter:
            gains_eval = gains_eval + float(jitter) * np.random.randn(len(gains_eval))
        if use_top_m is not None:
            gold = set(np.argsort(gains_eval)[::-1][:int(use_top_m)])
        else:
            thr = np.quantile(gains_eval, float(tau_q))
            gold = {i for i, g in enumerate(gains_eval) if g >= thr}

        rand_pred = np.random.permutation(len(gains_eval))
        per_model["Random"].append({
            f"P@{k}": precision_at_k(rand_pred, gold, k),
            f"R@{k}": recall_at_k(rand_pred, gold, k),
            f"MAP@{k}": ap_at_k(rand_pred, gold, k),
            f"MRR@{k}": mrr_at_k(rand_pred, gold, k),
            f"NDCG@{k}": ndcg_gain_at_k(rand_pred,gains_eval, k),
            f"BinaryNDCG@{k}": ndcg_binary_at_k(rand_pred, gold, k),
            f"BinaryDCG@{k}": dcg_binary_at_k(rand_pred, gold, k),
            f"Tau@{k}": kendall_tau_at_k(rand_pred, gains_eval, k),

        })

        T_proxy = text_features01(df, form, SENT_MODEL, k=PROXY_K)
        text_proxy = PROXY_W_REV * T_proxy[:, 1] + (1.0 - PROXY_W_REV) * T_proxy[:, 0]
        H_proxy = {**H, 'text': text_proxy}
        W_proxy_q = _active_weights(form, W_proxy, df)
        gains_proxy = aggregate_gains(H_proxy, W_proxy_q)
        pred_rule = np.argsort(gains_proxy)[::-1]

        corr = np.corrcoef(gains_proxy, gains_eval)[0,1]
        print(f"corr(proxy, eval)={corr:.3f}, |gold|={len(gold)}")

        assert any(abs(W_proxy[k]-W_eval[k]) > 1e-9 for k in W_eval), \
            "W_proxy et W_eval sont identiques -> RuleProxy deviendra oracle."
        per_model["RuleProxy"].append({
            f"P@{k}": precision_at_k(pred_rule, gold, k),
            f"R@{k}": recall_at_k(pred_rule, gold, k),
            f"MAP@{k}": ap_at_k(pred_rule, gold, k),
            f"MRR@{k}": mrr_at_k(pred_rule, gold, k),
            f"NDCG@{k}": ndcg_gain_at_k(pred_rule, gains_eval, k),
            f"BinaryNDCG@{k}": ndcg_binary_at_k(pred_rule, gold, k),
            f"BinaryDCG@{k}": dcg_binary_at_k(pred_rule, gold, k),
            f"Tau@{k}": kendall_tau_at_k(pred_rule, gains_eval, k),
        })  

        T_feat  = text_features01(df, form, SENT_MODEL, k=PROXY_K)
        Xq = pair_features(Zf, X_items, T_feat)
        for name, m in models.items():
            if hasattr(m, "predict_proba"):
                scores = m.predict_proba(Xq)[:, 1]
            elif hasattr(m, "decision_function"):
                scores = m.decision_function(Xq)
            else:
                scores = m.predict(Xq).astype(float)
            pred = np.argsort(scores)[::-1]
            per_model[name].append({
                f"P@{k}": precision_at_k(pred, gold, k),
                f"R@{k}": recall_at_k(pred, gold, k),
                f"MAP@{k}": ap_at_k(pred, gold, k),
                f"MRR@{k}": mrr_at_k(pred, gold, k),
                f"NDCG@{k}": ndcg_gain_at_k(pred, gains_eval, k),
                f"BinaryNDCG@{k}": ndcg_binary_at_k(pred, gold, k),
                f"BinaryDCG@{k}": dcg_binary_at_k(pred, gold, k),
                f"Tau@{k}": kendall_tau_at_k(pred, gains_eval, k),
            })

    rows, perq_rows = {}, []
    keep = keep = [f"P@{k}", f"R@{k}", f"MAP@{k}", f"MRR@{k}", f"NDCG@{k}", f"BinaryDCG@{k}", f"Tau@{k}",f"BinaryNDCG@{k}"]
    for name, lst in per_model.items():
        if not lst:
            continue
        dfm = pd.DataFrame(lst)
        tmp = dfm.copy(); tmp["model"] = name
        perq_rows.append(tmp)
        agg = {}
        for mname in keep:
            mean_, lo, hi = _bootstrap_ci(dfm[mname].values, n_boot=n_boot)
            agg[f"{mname}_mean"] = mean_
            agg[f"{mname}_lo95"] = lo
            agg[f"{mname}_hi95"] = hi
        rows[name] = agg

    summary = pd.DataFrame(rows).T.sort_values(f"NDCG@{k}_mean", ascending=False)
    per_query = pd.concat(perq_rows, axis=0, ignore_index=True) if perq_rows else pd.DataFrame()
    return summary, per_query

def plot_metric_bars(summary_df, metric, out_path):
    m_mean = f"{metric}_mean"
    m_lo = f"{metric}_lo95"
    m_hi = f"{metric}_hi95"
    if not {m_mean, m_lo, m_hi}.issubset(summary_df.columns):
        raise ValueError(f"Métriques manquantes dans summary_df pour {metric}")
    labels = summary_df.index.tolist()
    vals = summary_df[m_mean].values
    err_low = vals - summary_df[m_lo].values
    err_up  = summary_df[m_hi].values - vals

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), vals, yerr=[err_low, err_up], capsize=4)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(f"{metric} (moyenne ± IC95)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_scatter(summary_df, x_metric, y_metric, out_path):
    xm = f"{x_metric}_mean"
    ym = f"{y_metric}_mean"
    if not {xm, ym}.issubset(summary_df.columns):
        raise ValueError("Colonnes manquantes pour le scatter")
    plt.figure(figsize=(6, 5))
    xs = summary_df[xm].values
    ys = summary_df[ym].values
    plt.scatter(xs, ys)
    for i, label in enumerate(summary_df.index.tolist()):
        plt.annotate(label, (xs[i], ys[i]))
    plt.xlabel(x_metric); plt.ylabel(y_metric)
    plt.title(f"{y_metric} vs {x_metric}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# =========================
#  Helpers: grid-search
# =========================
from itertools import product

def _build_pointwise_for(forms_subset, preproc, df, X_items, tau_val):
    X, y, sw, qid = build_pointwise(forms_subset, preproc, df, X_items, SENT_MODEL, tau=tau_val)
    return X, y, sw

def _eval_ndcg5(forms_subset, preproc, df, X_items, model_name, model_obj, n_boot=50):
    summary, _ = eval_benchmark(
        forms_subset, preproc, df, X_items, SENT_MODEL, {model_name: model_obj},
        k=5, tau_q=0.85, use_top_m=10, jitter=1e-6, n_boot=n_boot
    )
    # retourne 0 si le modèle n'a pas de métrique (sécurité)
    col = "NDCG@5_mean"
    return float(summary.loc[model_name, col]) if (model_name in summary.index and col in summary.columns) else 0.0

def grid_search_linear_svc(forms_tr_inner, forms_va, preproc, df, X_items,
                           grid_tau=(0.50, 0.60, 0.65, 0.70),
                           grid_C=(0.25, 0.5, 1.0, 2.0, 4.0),
                           n_boot_eval=50, verbose=True):
    """
    Cherche les meilleurs (tau, C) pour LinearSVC (pointwise) selon NDCG@5 sur la validation.
    """
    best = {"score": -1.0, "tau": None, "C": None, "model": None}
    for tau_val, C in product(grid_tau, grid_C):
        # train set (interne)
        Xtr, ytr, swtr = _build_pointwise_for(forms_tr_inner, preproc, df, X_items, tau_val)
        mdl = LinearSVC(random_state=42, C=float(C))
        mdl.fit(Xtr, ytr)  # LinearSVC n'utilise pas sample_weight de façon stable selon versions

        # validation
        sc = _eval_ndcg5(forms_va, preproc, df, X_items, "LinearSVC", mdl, n_boot=n_boot_eval)
        if verbose:
            print(f"[GS LinearSVC] tau={tau_val:.2f}  C={C:<4} -> NDCG@5={sc:.4f}")
        if sc > best["score"]:
            best.update(score=sc, tau=tau_val, C=C, model=mdl)
    if verbose:
        print(f"[GS LinearSVC] BEST -> tau={best['tau']:.2f}  C={best['C']}  NDCG@5={best['score']:.4f}")
    return best


from contextlib import contextmanager

@contextmanager
def temp_text_params(w_rev=None, k=None):
    """
    Patch temporaire de score_text pour imposer w_rev / k pendant la grid-search,
    sans toucher aux signatures existantes.
    """
    orig = score_text
    _w = w_rev
    _k = k

    def patched(df, form, model, w_rev=0.5, w_desc=0.5, k=3, missing_cos=0.0):
        # force w_rev / k si fournis, et ajuste w_desc = 1 - w_rev
        w = _w if _w is not None else w_rev
        return orig(df, form, model,
                    w_rev=w, w_desc=(1.0 - w), k=(_k if _k is not None else k),
                    missing_cos=missing_cos)

    globals()['score_text'] = patched
    try:
        yield
    finally:
        globals()['score_text'] = orig

from itertools import product

def _build_pointwise_for(forms_subset, preproc, df, X_items, tau_val):
    X, y, sw, _ = build_pointwise(forms_subset, preproc, df, X_items, SENT_MODEL, tau=tau_val)
    return X, y, sw

def _eval_ndcg5(forms_subset, preproc, df, X_items, model_name, model_obj, n_boot=40):
    summary, _ = eval_benchmark(
        forms_subset, preproc, df, X_items, SENT_MODEL, {model_name: model_obj},
        k=5, tau_q=0.85, use_top_m=10, jitter=1e-6, n_boot=n_boot
    )
    col = "NDCG@5_mean"
    return float(summary.loc[model_name, col]) if (model_name in summary.index and col in summary.columns) else 0.0

# ---------- LinearSVC (pointwise) : (tau, C, loss, class_weight) + (w_rev, k) ----------
def grid_search_linear_svc(forms_tr_inner, forms_va, preproc, df, X_items,
                           grid_tau=(0.55, 0.60, 0.65),
                           grid_C=(0.25, 0.5, 1.0, 2.0),
                           grid_loss=("hinge", "squared_hinge"),
                           grid_cw=(None, "balanced"),
                           grid_text_k=(1, 3),
                           grid_text_wrev=(0.5, 0.7),
                           n_boot_eval=30, verbose=True):
    best = {"score": -1.0}
    for w_rev, k in product(grid_text_wrev, grid_text_k):
        with temp_text_params(w_rev=w_rev, k=k):
            for tau_val, C, loss, cw in product(grid_tau, grid_C, grid_loss, grid_cw):
                Xtr, ytr, swtr = _build_pointwise_for(forms_tr_inner, preproc, df, X_items, tau_val)
                mdl = LinearSVC(random_state=42, C=float(C), loss=loss, class_weight=cw)
                mdl.fit(Xtr, ytr)

                sc = _eval_ndcg5(forms_va, preproc, df, X_items, "LinearSVC", mdl, n_boot=n_boot_eval)
                if verbose:
                    print(f"[GS LinearSVC] tau={tau_val:.2f} C={C} loss={loss} cw={cw} w_rev={w_rev} k={k} -> NDCG@5={sc:.4f}")
                if sc > best["score"]:
                    best = {
                        "score": sc, "tau": tau_val, "C": C, "loss": loss, "class_weight": cw,
                        "w_rev": w_rev, "k": k, "model": mdl
                    }
    if verbose:
        print(f"[GS LinearSVC] BEST -> {best}")
    return best

# ---------- RandomForest : ajoute max_features, min_samples_* , class_weight ----------
def grid_search_rf(forms_tr_inner, forms_va, preproc, df, X_items,
                   tau_val,
                   grid_ne=(300, 600, 800),
                   grid_md=(None, 15, 25),
                   grid_msl=(1, 3, 5),
                   grid_mss=(2, 5, 10),
                   grid_mf=("sqrt", 0.6),
                   grid_cw=(None, "balanced", "balanced_subsample"),
                   n_boot_eval=25, verbose=True):
    best = {"score": -1.0}
    for ne, md, msl, mss, mf, cw in product(grid_ne, grid_md, grid_msl, grid_mss, grid_mf, grid_cw):
        Xtr, ytr, swtr = _build_pointwise_for(forms_tr_inner, preproc, df, X_items, tau_val)
        mdl = RandomForestClassifier(
            n_estimators=int(ne), max_depth=md, min_samples_leaf=int(msl),
            min_samples_split=int(mss), max_features=mf, class_weight=cw,
            random_state=42, n_jobs=-1
        )
        mdl.fit(Xtr, ytr, sample_weight=swtr)
        sc = _eval_ndcg5(forms_va, preproc, df, X_items, "RandomForest", mdl, n_boot=n_boot_eval)
        if verbose:
            print(f"[GS RF] ne={ne} md={md} msl={msl} mss={mss} mf={mf} cw={cw} -> NDCG@5={sc:.4f}")
        if sc > best["score"]:
            best = {"score": sc, "n_estimators": ne, "max_depth": md, "min_samples_leaf": msl,
                    "min_samples_split": mss, "max_features": mf, "class_weight": cw, "model": mdl}
    if verbose:
        print(f"[GS RF] BEST -> {best}")
    return best

# ---------- HistGradientBoosting : lr / leaf / min_leaf / l2 / max_bins ----------
def grid_search_hgb(forms_tr_inner, forms_va, preproc, df, X_items,
                    tau_val,
                    grid_lr=(0.03, 0.05, 0.10),
                    grid_leaf=(31, 63, 127),
                    grid_min_leaf=(20, 50),
                    grid_l2=(0.0, 1e-3, 1e-2),
                    grid_bins=(127, 255),
                    n_boot_eval=25, verbose=True):
    best = {"score": -1.0}
    for lr, leaf, min_leaf, l2, mb in product(grid_lr, grid_leaf, grid_min_leaf, grid_l2, grid_bins):
        Xtr, ytr, swtr = _build_pointwise_for(forms_tr_inner, preproc, df, X_items, tau_val)
        mdl = HistGradientBoostingClassifier(
            learning_rate=float(lr), max_leaf_nodes=int(leaf), min_samples_leaf=int(min_leaf),
            l2_regularization=float(l2), max_bins=int(mb),
            early_stopping=True, validation_fraction=0.15, random_state=42
        )
        mdl.fit(Xtr, ytr, sample_weight=swtr)
        sc = _eval_ndcg5(forms_va, preproc, df, X_items, "HistGB", mdl, n_boot=n_boot_eval)
        if verbose:
            print(f"[GS HGB] lr={lr} leaf={leaf} min_leaf={min_leaf} l2={l2} bins={mb} -> NDCG@5={sc:.4f}")
        if sc > best["score"]:
            best = {"score": sc, "learning_rate": lr, "max_leaf_nodes": leaf,
                    "min_samples_leaf": min_leaf, "l2_regularization": l2, "max_bins": mb,
                    "model": mdl}
    if verbose:
        print(f"[GS HGB] BEST -> {best}")
    return best

# ---------- Pairwise SVM : C + loss ----------
def grid_search_pair_svm(forms_tr_inner, forms_va, preproc, df, X_items,
                         grid_C=(0.25, 0.5, 1.0, 2.0),
                         grid_loss=("hinge", "squared_hinge"),
                         n_boot_eval=20, verbose=True):
    Xp, yp, wp = build_pairwise(forms_tr_inner, preproc, df, X_items, SENT_MODEL, W=W_proxy, top_m=10, bot_m=10)
    Xp2 = np.vstack([Xp, -Xp]); yp2 = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xp))]); wp2 = np.concatenate([wp, wp])

    best = {"score": -1.0}
    for C, loss in product(grid_C, grid_loss):
        mdl = make_pipeline(StandardScaler(), LinearSVC(C=float(C), loss=loss, random_state=42))
        mdl.fit(Xp2, yp2, linearsvc__sample_weight=wp2)
        sc = _eval_ndcg5(forms_va, preproc, df, X_items, "PairSVM", mdl, n_boot=n_boot_eval)
        if verbose:
            print(f"[GS PairSVM] C={C} loss={loss} -> NDCG@5={sc:.4f}")
        if sc > best["score"]:
            best = {"score": sc, "C": C, "loss": loss, "model": mdl}
    if verbose:
        print(f"[GS PairSVM] BEST -> {best}")
    return best

# ---------- Pairwise LogReg : C + class_weight ----------
def grid_search_pair_lr(forms_tr_inner, forms_va, preproc, df, X_items,
                        grid_C=(0.25, 0.5, 1.0, 2.0, 4.0),
                        grid_cw=(None, "balanced"),
                        n_boot_eval=20, verbose=True):
    Xp, yp, wp = build_pairwise(forms_tr_inner, preproc, df, X_items, SENT_MODEL, W=W_proxy, top_m=10, bot_m=10)
    Xp2 = np.vstack([Xp, -Xp]); yp2 = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xp))]); wp2 = np.concatenate([wp, wp])

    best = {"score": -1.0}
    for C, cw in product(grid_C, grid_cw):
        mdl = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", solver="liblinear", max_iter=500, random_state=42,
                               C=float(C), class_weight=cw)
        )
        mdl.fit(Xp2, yp2, logisticregression__sample_weight=wp2)
        sc = _eval_ndcg5(forms_va, preproc, df, X_items, "PairLogReg", mdl, n_boot=n_boot_eval)
        if verbose:
            print(f"[GS PairLogReg] C={C} cw={cw} -> NDCG@5={sc:.4f}")
        if sc > best["score"]:
            best = {"score": sc, "C": C, "class_weight": cw, "model": mdl}
    if verbose:
        print(f"[GS PairLogReg] BEST -> {best}")
    return best


def grid_search_lgbm_ranker(forms_tr_inner, forms_va, preproc, df, X_items,
                             grid_lr=(0.03, 0.05, 0.10),
                             grid_leaves=(31, 63, 127),
                             grid_min_child=(5, 10, 20),
                             grid_l2=(0.0, 1e-3, 1e-2),
                             grid_subsample=(1.0,0.8),
                             grid_colsample=(1.0,0.8),
                             n_estimators=300,
                             n_boot_eval=25):
    """
    Liste de paramètres raisonnable pour LambdaRank (listwise).
    On évalue via _eval_ndcg5 (qui reconstruit Xq et appelle model.predict).
    """
    if lgb is None:
        print("[LGBMRanker] LightGBM non installé -> skip listwise.")
        return None

    best = {"score": -1.0}
    # Construit train/valid listwise une seule fois
    Xtr, ytr, gtr = build_listwise(forms_tr_inner, preproc, df, X_items, SENT_MODEL, W=W_proxy)
    Xva, yva, gva = build_listwise(forms_va,       preproc, df, X_items, SENT_MODEL, W=W_proxy)

    for lr in grid_lr:
        for nl in grid_leaves:
            for mcs in grid_min_child:
                for reg in grid_l2:
                    for ss in grid_subsample:
                        for cs in grid_colsample:
                            mdl = lgb.LGBMRanker(
                                objective="lambdarank",
                                n_estimators=int(n_estimators),
                                learning_rate=float(lr),
                                num_leaves=int(nl),
                                min_child_samples=int(mcs),
                                reg_lambda=float(reg),
                                subsample=float(ss),
                                colsample_bytree=float(cs),
                                random_state=42,
                                min_split_gain=0.0, 
                                label_gain=[0, 1, 3, 7], 
                            )
                            mdl.fit(
                                Xtr, ytr,
                                group=gtr,
                                eval_set=[(Xva, yva)],
                                eval_group=[gva],
                                eval_at=[5],
                            )
                            # éval via notre pipeline (reconstruit Xq et utilise mdl.predict)
                            sc = _eval_ndcg5(forms_va, preproc, df, X_items, "LGBMRanker", mdl, n_boot=n_boot_eval)
                            print(f"[GS LGBM] lr={lr} leaves={nl} min_child={mcs} "
                                    f"l2={reg} subsample={ss} colsample={cs} -> NDCG@5={sc:.4f}")
                            if sc > best["score"]:
                                best = {
                                    "score": sc,
                                    "learning_rate": lr,
                                    "num_leaves": nl,
                                    "min_child_samples": mcs,
                                    "reg_lambda": reg,
                                    "subsample": ss,
                                    "colsample_bytree": cs,
                                    "n_estimators": n_estimators,
                                    "model": mdl
                                }

    print(f"[GS LGBM] BEST -> {best}")
    return best
    
# =========================================================
#                         MAIN
# =========================================================

def main_param():
    df = load_and_prepare_catalog()
    assert not df.empty, "Catalogue vide."
    print(df.columns)
    preproc = build_preproc_for_items(df)

    forms_csv = os.getenv("FORMS_CSV", "API/entrainement/forms_restaurants_dept37_single_cp.csv")
    if not Path(forms_csv).exists():
        return "Erreur, csv non trouvé"
    def forms_from_csv(path_csv: str, df_catalog: pd.DataFrame):
            df_forms = pd.read_csv(path_csv)
            out = []
            for _, r in df_forms.iterrows():
                f = {"type": "restaurant"}
                if 'price_level' in df_forms.columns:
                    val = r.get('price_level', None)
                    try:
                        if pd.notna(val) and str(val).strip() != '':
                            f['price_level'] = int(float(val))
                    except Exception:
                        pass
                if 'code_postal' in df_forms.columns:
                    cp = str(r.get('code_postal', '')).strip()
                    if cp and cp.lower() != 'nan':
                        f['code_postal'] = cp
                if 'open' in df_forms.columns:
                    o = str(r.get('open', '')).strip()
                    if o and o.lower() != 'nan':
                        f['open'] = o
                if 'options' in df_forms.columns:
                    opts_raw = r.get('options', '')
                    opts = []
                    if pd.notna(opts_raw):
                        s = str(opts_raw).strip()
                        if s.startswith('['):
                            try:
                                parsed = json.loads(s)
                                if isinstance(parsed, list):
                                    opts = [x for x in parsed if isinstance(x, str)]
                            except Exception:
                                pass
                        if not opts:
                            toks = [t.strip() for t in re.split(r'[;,]', s) if t.strip()]
                            opts = toks
                    if opts:
                        f['options'] = opts
                if 'description' in df_forms.columns:
                    d = r.get('description', '')
                    if pd.notna(d):
                        s = str(d).strip()
                        if s and s.lower() != 'nan':
                            f['description'] = s
                out.append(f)
            return out
    forms = forms_from_csv(forms_csv, df)

    X_items = preproc.fit_transform(df)
    X_items = X_items.toarray().astype(np.float32) if hasattr(X_items, "toarray") else np.asarray(X_items, dtype=np.float32)

    forms_tr, forms_te = train_test_split(list(forms), test_size=0.25, random_state=123)
    forms_tr_inner, forms_va = train_test_split(list(forms_tr), test_size=0.30, random_state=456)

    # ---------- Tuning pointwise (texte inclus via temp_text_params) ----------
    best_svc = grid_search_linear_svc(forms_tr_inner, forms_va, preproc, df, X_items,
                                      n_boot_eval=30, verbose=True)
    tau_best = 0.6
    text_w_rev_best = 0.5
    text_k_best = 3


    best_lgbm = grid_search_lgbm_ranker(forms_tr_inner, forms_va, preproc, df, X_items,
                                        n_boot_eval=20) if lgb is not None else None
    best_rf = grid_search_rf(forms_tr_inner, forms_va, preproc, df, X_items,
                                tau_val=tau_best, n_boot_eval=25, verbose=True)
    best_hgb = grid_search_hgb(forms_tr_inner, forms_va, preproc, df, X_items,
                                tau_val=tau_best, n_boot_eval=25, verbose=True)
    best_pair_svm = grid_search_pair_svm(forms_tr_inner, forms_va, preproc, df, X_items,
                                            n_boot_eval=20, verbose=True)
    best_pair_lr  = grid_search_pair_lr(forms_tr_inner, forms_va, preproc, df, X_items,
                                        n_boot_eval=20, verbose=True)


    # pointwise & pairwise existants...
    Xtr, ytr, swtr, _ = build_pointwise(forms_tr, preproc, df, X_items, SENT_MODEL, tau=tau_best)
    Xte, yte, swte, qte = build_pointwise(forms_te, preproc, df, X_items, SENT_MODEL, tau=tau_best)

    Xp, yp, wp = build_pairwise(forms_tr, preproc, df, X_items, SENT_MODEL, W=W_proxy, top_m=10, bot_m=10)
    Xp2 = np.vstack([Xp, -Xp]); yp2 = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xp))]); wp2 = np.concatenate([wp, wp])

    mdl_svm = LinearSVC(random_state=42, C=float(best_svc["C"]),
                        loss=best_svc["loss"], class_weight=best_svc["class_weight"])
    mdl_svm.fit(Xtr, ytr)

    mdl_rf = RandomForestClassifier(
        n_estimators=int(best_rf["n_estimators"]), max_depth=best_rf["max_depth"],
        min_samples_leaf=int(best_rf["min_samples_leaf"]), min_samples_split=int(best_rf["min_samples_split"]),
        max_features=best_rf["max_features"], class_weight=best_rf["class_weight"],
        random_state=42, n_jobs=-1
    )
    mdl_rf.fit(Xtr, ytr, sample_weight=swtr)

    mdl_hgb = HistGradientBoostingClassifier(
        learning_rate=float(best_hgb["learning_rate"]), max_leaf_nodes=int(best_hgb["max_leaf_nodes"]),
        min_samples_leaf=int(best_hgb["min_samples_leaf"]), l2_regularization=float(best_hgb["l2_regularization"]),
        max_bins=int(best_hgb["max_bins"]), early_stopping=True, validation_fraction=0.15, random_state=42
    )
    mdl_hgb.fit(Xtr, ytr, sample_weight=swtr)

    pair_svm = make_pipeline(StandardScaler(),
                                LinearSVC(C=float(best_pair_svm["C"]), loss=best_pair_svm["loss"], random_state=42))
    pair_svm.fit(Xp2, yp2, linearsvc__sample_weight=wp2)

    pair_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", solver="liblinear", max_iter=500, random_state=42,
                            C=float(best_pair_lr["C"]), class_weight=best_pair_lr["class_weight"])
    )
    pair_lr.fit(Xp2, yp2, logisticregression__sample_weight=wp2)

    # (NEW) ré-entraînement final listwise si dispo
    if lgb is not None and best_lgbm is not None:
        Xlw_tr, ylw_tr, glw_tr = build_listwise(forms_tr, preproc, df, X_items, SENT_MODEL, W=W_proxy)
        mdl_lgbm = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=int(best_lgbm["n_estimators"]),
            learning_rate=float(best_lgbm["learning_rate"]),
            num_leaves=int(best_lgbm["num_leaves"]),
            min_child_samples=int(best_lgbm["min_child_samples"]),
            reg_lambda=float(best_lgbm["reg_lambda"]),
            subsample=float(best_lgbm["subsample"]),
            colsample_bytree=float(best_lgbm["colsample_bytree"]),
            random_state=42,
            label_gain=[0, 1, 3, 7]
        )
        mdl_lgbm.fit(Xlw_tr, ylw_tr, group=glw_tr)
    else:
        mdl_lgbm = None

    # Évaluation finale (ajoute LGBMRanker si présent)
    models = {
        "LinearSVC": mdl_svm,
        "RandomForest": mdl_rf,
        "HistGB": mdl_hgb,
        "PairSVM": pair_svm,
        "PairLogReg": pair_lr,
    }
    if mdl_lgbm is not None:
        models["LGBMRanker"] = mdl_lgbm

    summary, per_query = eval_benchmark(
        forms_te, preproc, df, X_items, SENT_MODEL, models,
        k=5, tau_q=0.85, use_top_m=10, jitter=1e-6, n_boot=300
    )

    # ---------- Sauvegardes ----------
    outdir = _artifacts_dir() 
    KEEP = {
    "P@5_mean","P@5_lo95","P@5_hi95",
    "Recall@5_mean","Recall@5_lo95","Recall@5_hi95",
    "BinaryNDCG@5_mean","BinaryNDCG@5_lo95","BinaryNDCG@5_hi95",
    "DCG@5_mean","DCG@5_lo95","DCG@5_hi95",}
    
    summary_to_save = summary[[c for c in summary.columns if c in KEEP]].copy()
    summary_to_save.to_csv(outdir / "benchmark_summary_6.csv", index=True)
    per_query.to_csv(outdir / "benchmark_per_query_param_6.csv", index=False)

    Path("API/artifacts").mkdir(exist_ok=True, parents=True)

    to_dump = {
        "tau_best": float(tau_best),
        "text": {"w_rev": float(text_w_rev_best), "k": int(text_k_best)},
        "LinearSVC": {
            "C": float(best_svc["C"]),
            "loss": str(best_svc["loss"]),
            "class_weight": (None if best_svc["class_weight"] is None else str(best_svc["class_weight"]))
        },
        "RF": _jsonify_best(best_rf),
        "HistGB": _jsonify_best(best_hgb),
        "PairSVM": _jsonify_best(best_pair_svm),
        "PairLogReg": _jsonify_best(best_pair_lr),
    }
    # (NEW) paramètres LGBM
    if lgb is not None and best_lgbm is not None:
        to_dump["LGBMRanker"] = {
            "learning_rate": float(best_lgbm["learning_rate"]),
            "num_leaves": int(best_lgbm["num_leaves"]),
            "min_child_samples": int(best_lgbm["min_child_samples"]),
            "reg_lambda": float(best_lgbm["reg_lambda"]),
            "subsample": float(best_lgbm["subsample"]),
            "colsample_bytree": float(best_lgbm["colsample_bytree"]),
            "n_estimators": int(best_lgbm["n_estimators"])
        }

    with open("artifacts/best_params_6.json", "w", encoding="utf-8") as f:
        json.dump(to_dump, f, ensure_ascii=False, indent=2)

    print("\nFichiers générés dans ./artifacts :")
    print("- benchmark_summary_param.csv")
    print("- benchmark_per_query_param.csv")
    print("- best_params.json")


def main_entrainement():
    # 1) Charger et préparer le catalogue depuis les tables
    df = load_and_prepare_catalog()
    df = df.sort_values("id_etab").reset_index(drop=True)
    assert not df.empty, "Catalogue vide."

    # 2) Préproc des items (features "classiques" pour Zf & X_items)
    preproc = build_preproc_for_items(df)

    # 3) Jeu de formulaires
    forms_csv = os.getenv("FORMS_CSV", "API/entrainement/forms_restaurants_dept37_single_cp.csv")
    if not Path(forms_csv).exists():
        return "Erreur, csv non trouvé"

    def forms_from_csv(path_csv: str, df_catalog: pd.DataFrame):
        df_forms = pd.read_csv(path_csv)
        out = []
        for _, r in df_forms.iterrows():
            f = {"type": "restaurant"}
            if 'price_level' in df_forms.columns:
                val = r.get('price_level', None)
                try:
                    if pd.notna(val) and str(val).strip() != '':
                        f['price_level'] = int(float(val))
                except Exception:
                    pass
            if 'code_postal' in df_forms.columns:
                cp = str(r.get('code_postal', '')).strip()
                if cp and cp.lower() != 'nan':
                    f['code_postal'] = cp
            if 'open' in df_forms.columns:
                o = str(r.get('open', '')).strip()
                if o and o.lower() != 'nan':
                    f['open'] = o
            if 'options' in df_forms.columns:
                opts_raw = r.get('options', '')
                opts = []
                if pd.notna(opts_raw):
                    s = str(opts_raw).strip()
                    if s.startswith('['):
                        try:
                            parsed = json.loads(s)
                            if isinstance(parsed, list):
                                opts = [x for x in parsed if isinstance(x, str)]
                        except Exception:
                            pass
                    if not opts:
                        toks = [t.strip() for t in re.split(r'[;,]', s) if t.strip()]
                        opts = toks
                if opts:
                    f['options'] = opts
            if 'description' in df_forms.columns:
                d = r.get('description', '')
                if pd.notna(d):
                    s = str(d).strip()
                    if s and s.lower() != 'nan':
                        f['description'] = s
            out.append(f)
        return out

    forms = forms_from_csv(forms_csv, df)
    print("debut IA")

    # 4) Matrice d'items (pour pair_features)
    # >>> IMPORTANT : on fit le préproc ICI (train uniquement), et on le SAUVEGARDE ensuite.
    X_items = preproc.fit_transform(df)
    X_items = X_items.toarray().astype(np.float32) if hasattr(X_items, "toarray") else np.asarray(X_items, dtype=np.float32)
    print(f"[dims] X_items = {X_items.shape}")  # debug utile pour d vs N

    # Split train/test sur les formulaires
    forms_tr, forms_te = train_test_split(list(forms), test_size=0.25, random_state=123)

    # 5) Datasets LISTWISE (pour LambdaRank)
    #    On utilise la même logique que dans le param tuning
    Xlw_tr, ylw_tr, glw_tr = build_listwise(forms_tr, preproc, df, X_items, SENT_MODEL, W=W_proxy)
    Xlw_te, ylw_te, glw_te = build_listwise(forms_te, preproc, df, X_items, SENT_MODEL, W=W_proxy)

    # 6) Modèle : LGBMRanker (LambdaRank)
    mdl_lgbm = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        reg_lambda=0.01,
        subsample=0.9,
        colsample_bytree=1,
        random_state=42,
        label_gain=[0, 1, 3, 7],
    )
    mdl_lgbm.fit(
        Xlw_tr, ylw_tr,
        group=glw_tr,
        eval_set=[(Xlw_te, ylw_te)],
        eval_group=[glw_te],
        eval_at=[5],
    )

    # 7) Évaluation ranking (pipeline commun)
    models = {"LGBMRanker": mdl_lgbm}
    summary, per_query = eval_benchmark(
        forms_te, preproc, df, X_items, SENT_MODEL, models,
        k=5, tau_q=0.85, use_top_m=10, jitter=1e-6, n_boot=300
    )
    print("\n=== Résumé (moyennes + IC95) ===")
    print(summary)

    # 8) Sauvegardes
    outdir = _artifacts_dir()
    KEEP = {
    "P@5_mean","P@5_lo95","P@5_hi95",
    "Recall@5_mean","Recall@5_lo95","Recall@5_hi95",
    "BinaryNDCG@5_mean","BinaryNDCG@5_lo95","BinaryNDCG@5_hi95",
    "DCG@5_mean","DCG@5_lo95","DCG@5_hi95",}
    
    summary_to_save = summary[[c for c in summary.columns if c in KEEP]].copy()
    summary_to_save.to_csv(outdir / "benchmark_summary.csv", index=True)
    per_query.to_csv(outdir / "benchmark_per_query.csv", index=False)

    preproc_path = outdir / "preproc_items1.joblib"
    model_path   = outdir / "rank_model1.joblib"

    print(f"[save] preproc -> {preproc_path.resolve()}")
    joblib.dump(preproc, preproc_path, compress=("xz", 3))

    rank_model = FallbackRanker(mdl_lgbm , tie_cols=(-2, -1))
    print(f"[save] rank model ({type(rank_model)}) -> {model_path.resolve()}")
    try:
        joblib.dump(rank_model, model_path, compress=("xz", 3))
        print("[save] OK rank_model.joblib")
    except Exception as e:
        import traceback
        print("[save][ERROR] Impossible d'écrire rank_model.joblib :", e)
        traceback.print_exc()

    print("\nFichiers générés dans", outdir.resolve())
    print("\nFichiers générés")


def _dispatch(mode: str):
    mode = (mode or "").lower()
    if mode in {"param", "tuning", "grid"}:
        return main_param()
    if mode in {"train", "entrainement"}:
        return main_entrainement()
    # fallback (compat)
    print("Mode inconnu. Utilise --mode param | train")
    return main_entrainement()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner E3: tuning ou entraînement")
    parser.add_argument(
        "-m", "--mode",
        choices=["param", "train"],
        help="param = recherche d'hyperparamètres ; train = entraînement simple"
    )
    args, unknown = parser.parse_known_args()

    # Priorité à l'argument CLI ; sinon variable d'env ; sinon 'train'
    mode = args.mode or os.getenv("RUN_MODE", "train")
    _dispatch(mode)