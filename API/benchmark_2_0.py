import pandas as pd
import numpy as np
import time
import json 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pgeocode
import re
import unicodedata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from pathlib import Path
import joblib
import os

try:
    from . import CRUD
except:
    import CRUD

"""model = SentenceTransformer('BAAI/bge-m3')"""

W_proxy = {'price':0.18,'rating':0.14,'options':0.14,'text':0.28,'city':0.04,'open':0.04}
W_eval  = {'price':0.12,'rating':0.18,'options':0.12,'text':0.22,'city':0.20,'open':0.16}
tau = 0.60

def _is_nanlike(x):
    return isinstance(x, float) and np.isnan(x)

def _text_or_empty(x):
    if x is None or _is_nanlike(x):
        return ''
    return str(x).strip()

def fget(form, key, default=None):
    """Accès robuste à un champ de form (dict, pandas Series, namedtuple...)."""
    if isinstance(form, dict):
        return form.get(key, default)
    if isinstance(form, pd.Series):
        return form.get(key, default)
    return getattr(form, key, default)

def first_code_postal_5digits(pcs):
    if isinstance(pcs, pd.Series): pcs = pcs.dropna().tolist()
    if isinstance(pcs, np.ndarray): pcs = pcs.tolist()
    values = pcs if isinstance(pcs, (list, tuple, set)) else [pcs]

    for v in values:
        if v is None: 
            continue
        m = re.search(r'\b\d{5}\b', str(v))
        if m:
            return m.group(0)
    return ''

def form_to_row(form, df_catalog):
    row = {c: np.nan for c in df_catalog.columns}

    if 'type' in df_catalog.columns:
        row['type'] = fget(form, 'type', '')

    txt = fget(form, 'description', '') or ''
    if 'editorialSummary_text' in df_catalog.columns:
        row['editorialSummary_text'] = txt
    elif 'description' in df_catalog.columns:
        row['description'] = txt

    if 'priceLevel' in df_catalog.columns:
        lvl = fget(form, 'price_level', None)
        if lvl is None or (isinstance(lvl, float) and np.isnan(lvl)):
            row['priceLevel'] = np.nan
        else:
            if np.issubdtype(df_catalog['priceLevel'].dtype, np.number):
                row['priceLevel'] = float(lvl)
            else:
                mapping = {1:'$', 2:'$$', 3:'$$$', 4:'$$$$'}
                row['priceLevel'] = mapping.get(int(lvl), '')

    if 'code_postal' in df_catalog.columns:
        pcs = fget(form, 'code_postal', None)
        row['code_postal'] = first_code_postal_5digits(pcs)

    opts = fget(form, 'options', [])
    if isinstance(opts, str) and opts.strip():
        opts = [x.strip() for x in re.split(r'[;,]', opts)]
    if isinstance(opts, (list, tuple, set)):
        for c in opts:
            if c in df_catalog.columns:
                row[c] = True

    for c in ['rating','start_price','end_price','mean_review_rating']:
        if c in df_catalog.columns:
            row[c] = 0.0

    if 'review_list' in df_catalog.columns:
        row['review_list'] = []

    return pd.DataFrame([row])[df_catalog.columns]

def _normalize_multivalue(value):
    if value is None:
        return []
    if isinstance(value, pd.Series):
        value = value.dropna().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        seq = value
    else:
        # si c'est une chaîne de type "a, b; c", on découpe
        s = str(value)
        if any(sep in s for sep in [",", ";"]):
            seq = re.split(r"[;,]\s*", s)
        else:
            seq = [value]
    out = []
    for v in seq:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out 

def h_price_vector(df, form):
    if 'priceLevel' not in df.columns:
        return np.ones(len(df), dtype=float)

    lvl_f = fget(form, 'price_level', None)
    lvl_ok = None
    if isinstance(lvl_f, (int, float, np.floating)) and not _is_nanlike(lvl_f):
        lvl_ok = float(lvl_f)

    if lvl_ok is None:
        return np.ones(len(df), dtype=float)

    pl = df['priceLevel'].astype(float)
    diff = (pl.fillna(lvl_ok) - lvl_ok).abs()
    return (1.0 - (diff / 3.0)).clip(0.0, 1.0).to_numpy(dtype=float)

def h_rating_vector(df, alpha=20.0):
    r = df['rating'].astype(float).fillna(2.5) if 'rating' in df.columns else pd.Series(2.5, index=df.index)
    mu = float(r.mean()) if r.notna().any() else 0.0

    if 'review_list' in df.columns:
        n = df['review_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).astype(float)
    else:
        n = pd.Series(1.0, index=df.index)

    r_star = (n * r + alpha * mu) / (n + alpha)
    return np.clip(r_star / 5.0, 0.0, 1.0).to_numpy(dtype=float)

def h_city_vector(df, form):
    if 'code_postal' not in df.columns:
        return np.ones(len(df), dtype=float)

    pcs_raw = fget(form, 'code_postal', None)
    pcs = _normalize_multivalue(pcs_raw)
    if not pcs:
        return np.ones(len(df), dtype=float)

    s = df['code_postal'].astype(str).str.strip()
    return s.isin(pcs).astype(float).to_numpy(dtype=float)

def _extract_requested_options(form, df_catalog):
    opts = fget(form, 'options', None)
    if isinstance(opts, str) and opts.strip():
        toks = [t.strip() for t in re.split(r'[;,]\s*', opts.strip()) if t.strip()]
        return [t for t in toks if t in df_catalog.columns]

    if isinstance(opts, (list, tuple, set)):
        toks = [str(o).strip() for o in opts if str(o).strip()]
        return [t for t in toks if t in df_catalog.columns]

    out = []
    for c in df_catalog.columns:
        if df_catalog[c].dtype == bool:
            v = fget(form, c, None)
            if isinstance(v, (bool, np.bool_)):
                if v:
                    out.append(c)
            elif isinstance(v, (int, float)) and int(v) == 1:
                out.append(c)
            elif isinstance(v, str) and v.strip().lower() in ('1', 'true', 'vrai', 'yes', 'oui'):
                out.append(c)
    return out

def h_opts_vector(df, form):
    req = _extract_requested_options(form, df)
    if not req:
        return np.ones(len(df), dtype=float)

    sub = df[req].copy()
    for c in req:
        sub[c] = sub[c].map(lambda x: bool(x) if not pd.isna(x) else False)
    M = sub.astype(int).to_numpy()
    return M.mean(axis=1).astype(float)

def h_open_vector(df, form, unknown_value=1.0):
    col = fget(form, 'open', None)
    if not col or not isinstance(col, str):
        return np.ones(len(df), dtype=float)

    col = col.strip()
    if not col:
        return np.ones(len(df), dtype=float)

    cols = [c for c in df.columns if c.startswith(col)]
    if not cols and col not in df.columns:
        return np.ones(len(df), dtype=float)

    candidate_cols = cols if cols else [col]
    M = df[candidate_cols].fillna(0).astype(int)
    has_any_open_info = (M.sum(axis=1) > 0).to_numpy()

    if col in df.columns:
        v = df[col].fillna(0).astype(int).to_numpy().astype(float)
    else:
        v = (M.max(axis=1)).to_numpy(dtype=float)
    return np.where(has_any_open_info, v, float(unknown_value))

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
    q = _text_or_empty(fget(form, 'description', None))
    if not q:
        return np.ones(len(df), dtype=float)
    z = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
    N = len(df); out = np.empty(N, dtype=float)

    desc_list = df.get("desc_embed")
    cos_desc = np.array([float(v @ z) if isinstance(v, np.ndarray) and v.ndim==1 else None
                         for v in (desc_list if desc_list is not None else [None]*N)], dtype=object)

    rev_list = df.get("rev_embeds", [None]*N)
    for i in range(N):
        cos_r = topk_mean_cosine(rev_list[i], z, k=k) if rev_list is not None else None
        cos_d = cos_desc[i]
        if (cos_d is not None) and (cos_r is not None): c = w_rev*cos_r + w_desc*cos_d
        elif cos_r is not None:                         c = cos_r
        elif cos_d is not None:                         c = cos_d
        else:                                           c = float(missing_cos)
        out[i] = (c + 1.0) / 2.0
    return out

def score_func(df, form, model):
    score={}
    score["price"] = h_price_vector(df, form)
    score["rating"] = h_rating_vector(df, alpha=20.0)
    score["city"] = h_city_vector(df, form)
    score["options"] = h_opts_vector(df, form)
    score["open"] = h_open_vector(df,form, unknown_value=1.0)
    score["text"] = score_text(df, form, model,w_rev=0.6, w_desc=0.4, k=3, missing_cos=0.0)
    return score

def aggregate_gains(H, weights):
    if not H:
        return np.array([], dtype=float)

    keys = [k for k in ("price","rating","options","text","city","open") if k in H] or list(H.keys())
    N = len(np.asarray(H[keys[0]], dtype=float))

    W = np.array([float(weights.get(k, 1.0)) for k in keys], dtype=float)
    if W.sum() == 0:
        W[:] = 1.0
    W = W / W.sum()

    gains = np.zeros(N, dtype=float)
    for k, w in zip(keys, W):
        v = np.asarray(H[k], dtype=float)
        gains += w * v
    return gains

def pair_features_df(H, df= None, include= ("price","rating","options","text","city","open"), as_distance= False,
    id_col = "id_etab", prefix = "feat_", gains_weights = None, add_gains = True):
    if not H:
        return pd.DataFrame()

    used = [k for k in include if k in H] or list(H.keys())

    N = len(np.asarray(H[used[0]], dtype=float))
    index = df.index if isinstance(df, pd.DataFrame) else pd.RangeIndex(N)

    data = {}

    if isinstance(df, pd.DataFrame) and id_col in df.columns:
        data[id_col] = df[id_col].values

    for k in used:
        v = np.asarray(H[k], dtype=float)
        data[f"{prefix}{k}"] = (1.0 - v) if as_distance else v

    out = pd.DataFrame(data, index=index)

    if add_gains and gains_weights is not None:
        keys_in = [k for k in used if f"{prefix}{k}" in out.columns and k in gains_weights]
        if keys_in:
            W = np.array([float(gains_weights[k]) for k in keys_in], dtype=float)
            W = np.ones_like(W) if W.sum() == 0 else W / W.sum()
            M = out[[f"{prefix}{k}" for k in keys_in]].to_numpy(dtype=float)
            out["gain"] = (M * W).sum(axis=1)
    return out

def _iter_forms(forms):
    if isinstance(forms, pd.DataFrame):
        for _, row in forms.iterrows():
            yield row
    elif isinstance(forms, (list, tuple)):
        for f in forms:
            yield f
    elif isinstance(forms, dict) or isinstance(forms, pd.Series):
        yield forms
    else:
        yield forms

def compute_form_embed(form, sent_model):
    q = _text_or_empty(fget(form, 'description', None))
    if not q:
        return None
    z = sent_model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    return z.astype(np.float32)

def pick_anchors_from_df(df, n=8):
    vecs = [v for v in df.get('desc_embed', []) if isinstance(v, np.ndarray) and v.ndim == 1]
    if not vecs:
        return None
    vecs = np.stack(vecs).astype(np.float32)
    idx = np.linspace(0, len(vecs)-1, num=min(n, len(vecs)), dtype=int)
    return vecs[idx]

def anchor_cos_features(zf, anchors):
    if zf is None or anchors is None:
        return {}
    sims = anchors @ zf 
    return {f"q_anchor_{i}": float((s + 1.0)/2.0) for i, s in enumerate(sims)}

def build_item_features_df(df, form, sent_model, include_query_consts=False, anchors=None):
    H = score_func(df, form, sent_model)

    data = {
        "feat_price":   np.asarray(H["price"],   float),
        "feat_rating":  np.asarray(H["rating"],  float),
        "feat_options": np.asarray(H["options"], float),
        "feat_text":    np.asarray(H["text"],    float),
        "feat_city":    np.asarray(H["city"],    float),
        "feat_open":    np.asarray(H["open"],    float),
    }

    lvl = fget(form, 'price_level', np.nan)
    if "priceLevel" in df.columns:
        lvl_num = (np.nan if (lvl is None or (isinstance(lvl, float) and np.isnan(lvl)))
                   else float(lvl))
        data["f_price_absdiff"] = (
            np.abs(df["priceLevel"].astype(float) - (0.0 if np.isnan(lvl_num) else lvl_num))
            .fillna(0.0).to_numpy() / 3.0
        )

    req_opts = _extract_requested_options(form, df)
    data["f_req_count"] = np.full(len(df), float(len(req_opts) if req_opts else 0.0))

    zf = compute_form_embed(form, sent_model)
    if "desc_embed" in df.columns and zf is not None:
        v = (
            df["desc_embed"]
              .apply(lambda e: float(e @ zf) if isinstance(e, np.ndarray) and e.ndim == 1 else 0.0)
              .apply(lambda c: (c + 1.0) / 2.0)
              .to_numpy()
        )
    else:
        v = np.zeros(len(df), dtype=float) 
    data["f_text_desc_cos"] = v

    for c in ["rating", "priceLevel", "start_price", "latitude", "longitude"]:
        if c in df.columns:
            data[f"raw_{c}"] = df[c].astype(float).to_numpy()

    if include_query_consts:
        data["form_price_level"] = np.full(
            len(df),
            (np.nan if (lvl is None or (isinstance(lvl, float) and np.isnan(lvl))) else float(lvl))
        )
        if anchors is not None:
            K = anchors.shape[0] if isinstance(anchors, np.ndarray) else len(anchors)
            if zf is not None:
                sims = (anchors @ zf).astype(float)  # cos [-1,1]
                sims = (sims + 1.0) / 2.0           # map to [0,1]
            else:
                sims = np.full(K, 0.0, dtype=float) # neutre si pas de texte
            for i in range(K):
                data[f"q_anchor_{i}"] = np.full(len(df), sims[i], dtype=float)

    X_df = pd.DataFrame(data, index=df.index)
    if "id_etab" in df.columns:
        X_df["id_etab"] = df["id_etab"].values

    gains = aggregate_gains(H, W_proxy)
    return X_df, gains

def build_pointwise_from_X_df(X_df, gains, tau=0.60, drop_cols=("id_etab",)):
    """
    Pointwise pour UN formulaire déjà transformé en features par item.
    - X_df : DataFrame des features (une ligne par item)
    - gains: array-like de scores proxy alignés sur X_df
    - tau  : seuil pour le label binaire
    - drop_cols : colonnes à retirer des features (ex. identifiants)
    Retourne: X (ndarray), y (0/1), sw (poids), cols (noms de colonnes utilisées)
    """
    if X_df is None or len(X_df) == 0:
        return np.zeros((0,0), dtype=np.float32), np.array([], dtype=int), np.array([], dtype=float), []

    gains = np.asarray(gains, dtype=float)
    cols = [c for c in X_df.columns if c not in drop_cols]
    X = X_df[cols].to_numpy(dtype=float)

    y = (gains >= float(tau)).astype(int)
    sw = np.abs(gains - float(tau)) + 1e-3  
    return X.astype(np.float32), y, sw, cols

def build_pairwise_from_X_df(X_df, gains, top_m=10, bot_m=10, drop_cols=("id_etab",)):
    cols = [c for c in X_df.columns if c not in drop_cols]  
    order = np.argsort(gains)
    pos_idx = order[::-1][:top_m]
    neg_idx = order[:bot_m]

    diffs, wts = [], []
    for ip in pos_idx:
        for ineg in neg_idx:
            diffs.append(X_df.iloc[ip][cols].to_numpy(dtype=float) - X_df.iloc[ineg][cols].to_numpy(dtype=float))
            wts.append(float(gains[ip] - gains[ineg]))

    X_pairs = np.vstack(diffs).astype(np.float32) if diffs else np.zeros((0, len(cols)), dtype=np.float32)
    y = np.ones(len(wts), dtype=int)
    w = np.asarray(wts, float)
    return X_pairs, y, w, cols

def build_pointwise_dataset(forms, df, sent_model,
                            tau=0.60,
                            include_query_consts=False,
                            anchors=None,
                            id_col="id_etab"):
    """
    Pointwise pour PLUSIEURS formulaires.
    - forms : dict, liste de dicts, ou DataFrame de formulaires bruts (non préprocessés)
    - df    : catalogue items (avec colonnes, options, embeddings desc/rev, etc.)
    - sent_model : modèle d'embedding de texte (SentenceTransformer)
    - tau   : seuil binaire
    - include_query_consts : inclure quelques scalaires constants du form (petit K) dans X_df
    - anchors : matrice (K, d) d’ancres pour condenser l’embedding de la requête. Si None, on en pioche depuis df.
    - id_col : nom de la colonne identifiant les items (si présente, retirée des features)

    Retourne: X, y, sw, qid, cols
      - X   : empilement de toutes les lignes items pour tous les formulaires
      - y   : labels 0/1 par ligne
      - sw  : poids d’exemple
      - qid : id de requête (même valeur pour toutes les lignes issues du même form)
      - cols: colonnes de features utilisées
    """
    if anchors is None:
        anchors = pick_anchors_from_df(df, n=8)

    X_all, y_all, sw_all, qid_all = [], [], [], []
    cols_ref = None

    n_items = len(df)
    for qid, form in enumerate(_iter_forms(forms)):
        X_df, gains = build_item_features_df(df, form, sent_model,
            include_query_consts=include_query_consts,anchors=anchors)

        X, y, sw, cols = build_pointwise_from_X_df(
            X_df, gains, tau=tau, drop_cols=(id_col,)
        )
        if cols_ref is None:
            cols_ref = cols

        if X.size:
            X_all.append(X)
            y_all.append(y)
            sw_all.append(sw)
            qid_all.append(np.full(n_items, qid, dtype=int))

    if not X_all:
        return (np.zeros((0, 0), dtype=np.float32),np.array([], dtype=int),
                np.array([], dtype=float),np.array([], dtype=int),(cols_ref or []))

    X = np.vstack(X_all).astype(np.float32)
    y = np.concatenate(y_all)
    sw = np.concatenate(sw_all).astype(float)
    qid = np.concatenate(qid_all).astype(int)
    return X, y, sw, qid, cols_ref


def make_preproc_final():
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale",  StandardScaler()) 
    ])

def load_forms_csv(path_csv: str):
    return pd.read_csv(path_csv)

def train_pointwise_from_csv(forms_csv: str,out_path: str | None = None,clf_name: str = "LinearSVC"):
    df_items = CRUD.load_df()           
    forms_df = load_forms_csv(forms_csv)    

    anchors = pick_anchors_from_df(df_items, n=8)

    X, y, sw, qid, cols = build_pointwise_dataset(forms=forms_df,df=df_items,sent_model=model,           
        tau=tau,include_query_consts=True,anchors=anchors,id_col="id_etab",)

    preproc = make_preproc_final() 

    if clf_name == "LinearSVC":
        clf = LinearSVC()
    elif clf_name == "LogReg":
        clf = LogisticRegression(max_iter=2000)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier()

    pipe = Pipeline([("preproc", preproc), ("clf", clf)])
    pipe.fit(X, y, **({"clf__sample_weight": sw} if hasattr(clf, "fit") else {}))

    if out_path is None:
        out_path = os.getenv(
            "RANK_MODEL_PATH",
            str(Path("artifacts") / "linear_svc_pointwise.joblib")
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"[train] modèle sauvegardé -> {out_path}")
    return out_path, cols

if __name__ == "__main__":
    train_pointwise_from_csv("forms_restaurants_dept37_single_cp.csv")