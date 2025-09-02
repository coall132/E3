# API/features_runtime.py
import re, math
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

# =========================
#   Petits utilitaires
# =========================
def fget(form, key, default=None):
    if isinstance(form, dict):
        return form.get(key, default)
    if isinstance(form, pd.Series):
        return form.get(key, default)
    return getattr(form, key, default)

def sanitize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df.rename(columns=lambda c: re.sub(r'^"+|"+$', '', c), inplace=True)
    return df

def determine_price_level_row(row):
    sp = row.get('start_price', np.nan)
    if pd.notna(sp):
        try:
            price = float(sp)
        except Exception:
            return np.nan
        if price < 15: return 1
        elif price <= 20: return 2
        else: return 3
    return row.get('priceLevel', np.nan)

def _first_code_postal(pcs):
    if isinstance(pcs, pd.Series): pcs = pcs.dropna().tolist()
    if isinstance(pcs, np.ndarray): pcs = pcs.tolist()
    if isinstance(pcs, (list, tuple, set)):
        for v in pcs:
            if v is None: continue
            s = str(v).strip()
            if s: return s
        return ''
    if pcs is None: return ''
    return str(pcs).strip()

# =========================
#   Scoring texte & signaux
# =========================
def topk_mean_cosine(mat_or_list, z, k=3):
    if mat_or_list is None: return None
    if isinstance(mat_or_list, list):
        if len(mat_or_list) == 0: return None
        M = np.vstack([np.asarray(v, dtype=np.float32) for v in mat_or_list])
    else:
        M = np.asarray(mat_or_list, dtype=np.float32)
        if M.ndim == 1: M = M[None, :]
    if M.size == 0: return None
    sims = M @ z.astype(np.float32)
    k = min(int(k), len(sims))
    if k <= 0: return None
    idx = np.argpartition(sims, -k)[-k:]
    return float(np.mean(sims[idx]))

def score_text(df, form, model, w_rev=0.6, w_desc=0.4, k=3, missing_cos=0.0):
    q = (fget(form, 'description', '') or '').strip()
    if not q:
        return np.ones(len(df), dtype=float)
    z = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
    N = len(df); out = np.empty(N, dtype=float)

    desc_list = df.get("desc_embed")
    cos_desc = np.array([
        float(v @ z) if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 0 else None
        for v in (desc_list if desc_list is not None else [None]*N)
    ], dtype=object)

    # marge du top desc
    if any(c is not None for c in cos_desc):
        cosd = np.array([(-1.0 if c is None else float(c)) for c in cos_desc], dtype=float)
        top = int(np.argmax(cosd))
        margin = float(cosd[top] - np.partition(cosd, -2)[-2]) if len(cosd) >= 2 else 1.0
    else:
        margin = 0.0

    rev_list = df.get("rev_embeds", [None]*N)

    for i in range(N):
        cos_r = topk_mean_cosine(rev_list[i], z, k=k) if rev_list is not None else None
        cos_d = cos_desc[i]
        if margin >= 0.10 and cos_d is not None:
            c = cos_d
        else:
            if (cos_d is not None) and (cos_r is not None):
                c = w_rev*cos_r + (1.0 - w_rev)*cos_d
            elif cos_r is not None:
                c = cos_r
            elif cos_d is not None:
                c = cos_d
            else:
                c = float(missing_cos)
        out[i] = (c + 1.0) / 2.0
    return out

def h_price_vector_simple(df, form):
    lvl_f = fget(form, 'price_level', None)
    if lvl_f is None:
        return np.ones(len(df), dtype=float)
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
        return np.ones(len(df), dtype=float)
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
        return np.ones(len(df), dtype=float)
    sub = df[req].fillna(False).astype(int).to_numpy()
    return sub.mean(axis=1).astype(float)

def h_open_vector(df, form, unknown_value=1.0):
    col = fget(form, 'open', None)
    if not col or col not in df.columns:
        return np.ones(len(df), dtype=float)
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
        "open":    h_open_vector(df, form, unknown_value=1.0),
        "text":    score_text(df, form, model, w_rev=0.6, w_desc=0.4, k=3, missing_cos=0.0),
    }

def aggregate_gains(H, weights):
    keys = list(H)
    W = np.array([float(weights[k]) for k in keys], dtype=float)
    gains = np.zeros_like(np.asarray(H[keys[0]], dtype=float), dtype=float)
    for k, w in zip(keys, W):
        gains += w * np.asarray(H[k], dtype=float)
    return gains / W.sum()

# pondérations cohérentes avec ton entraînement
W_proxy = {'price':0.18,'rating':0.14,'options':0.14,'text':0.28,'city':0.04,'open':0.04}
W_eval  = {'price':0.12,'rating':0.18,'options':0.12,'text':0.22,'city':0.20,'open':0.16}

# =========================
#   Features pour l’API
# =========================
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

def pair_features(Zf, H, X_items, diff_scale=0.05):
    diff = np.abs(X_items - Zf.reshape(1, -1)) * float(diff_scale)
    cos  = np.asarray(H['text'], float)[:, None]
    return np.hstack([diff, cos]).astype(np.float32)

def _toint_safe(X):
    # robustifier le cast (évite ton crash actuel)
    if hasattr(X, "fillna"):   # DataFrame
        return X.fillna(0).astype(np.int32)
    return np.nan_to_num(X, nan=0.0).astype(np.int32)

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

    num_pipe  = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                          ('scale', StandardScaler())])

    bool_pipe = Pipeline([
        ('toint', FunctionTransformer(_toint_safe)),
        ('onehot', OneHotEncoder(categories=bool_categories,
                                 drop='if_binary', handle_unknown='ignore', sparse_output=True))
    ])

    lev_pipe  = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                          ('scale', StandardScaler())])

    return ColumnTransformer(
        transformers=[
            ("num",  num_pipe,  [c for c in NUM_COLS if c in df.columns]),
            ("bool", bool_pipe, [c for c in BOOL_COLS if c in df.columns]),
            ("lev",  lev_pipe,  [lev] if lev in df.columns else []),
        ],
        remainder="drop",
    )

# (optionnel) petites aides ancres/embeds si tu veux conserver le champ q_anchor_* un jour
def pick_anchors_from_df(df, n=8):
    vecs = [v for v in df.get('desc_embed', []) if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 0]
    if not vecs:
        return None
    lengths = [v.size for v in vecs]
    mode_len = max(set(lengths), key=lengths.count)
    vecs = [v for v in vecs if v.size == mode_len]
    if not vecs:
        return None
    M = np.stack(vecs).astype(np.float32)
    idx = np.linspace(0, len(vecs) - 1, num=min(n, len(vecs)), dtype=int)
    return M[idx]

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

    zf = None
    q = (fget(form, 'description', '') or '').strip()
    if q:
        zf = sent_model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
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

    X_df = pd.DataFrame(data, index=df.index)
    if "id_etab" in df.columns:
        X_df["id_etab"] = df["id_etab"].values

    gains = aggregate_gains(H, W_proxy)
    return X_df, gains
