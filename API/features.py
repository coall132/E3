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

def _cos01_safe(v: np.ndarray, z: np.ndarray) -> float:
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
                cos_d[i] = _cos01_safe(v, z)

    # cos top-k sur les reviews
    cos_r = np.full(N, np.nan, dtype=np.float32)
    rev_list = df.get('rev_embeds', None)
    if rev_list is not None:
        for i, L in enumerate(rev_list):
            if isinstance(L, list) and len(L) > 0:
                vals = [
                    _cos01_safe(r, z)
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
    """
    Construit les features modèle:
      - Zf: vecteur du formulaire (d,)
      - X_items: matrice items (N, d) (dense ou sparse)
      - T: features texte brutes (N, 2) dans [0,1] (cos_desc01, cos_rev01)
    Retour: (N, d + 2)
    """
    Zf = np.asarray(Zf, dtype=np.float32).ravel()
    if hasattr(X_items, "toarray"):
        X_items = X_items.toarray()
    X_items = np.asarray(X_items, dtype=np.float32)
    if X_items.ndim != 2:
        raise ValueError(f"X_items doit être 2D, reçu shape={X_items.shape}")

    T = np.asarray(T, dtype=np.float32)
    if T.ndim == 1:
        T = T[:, None]
    if T.shape[0] != X_items.shape[0]:
        raise ValueError(f"Mauvais N entre X_items (N={X_items.shape[0]}) et T (N={T.shape[0]})")

    d = X_items.shape[1]
    if Zf.shape[0] != d:
        raise ValueError(f"Zf dim={Zf.shape[0]} incompatible avec X_items dim={d}")

    diff = np.abs(X_items - Zf.reshape(1, -1)) * float(diff_scale)
    return np.hstack([diff, T]).astype(np.float32)

def pair_features(Zf, X_items, T, diff_scale=0.05):
    diff = np.abs(X_items - Zf.reshape(1, -1)) * float(diff_scale)  # (N, d)
    T = np.asarray(T, dtype=np.float32)                             # (N, 2) = [cos_desc01, cos_rev01]
    return np.hstack([diff, T]).astype(np.float32)

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
        "feat_price":   np.asarray(H["price"],   np.float32),
        "feat_rating":  np.asarray(H["rating"],  np.float32),
        "feat_options": np.asarray(H["options"], np.float32),
        "feat_text":    np.asarray(H["text"],    np.float32),  # proxy texte (rev/desc mixé)
        "feat_city":    np.asarray(H["city"],    np.float32),
        "feat_open":    np.asarray(H["open"],    np.float32),
    }

    # écart absolu au niveau de prix (normalisé /3)
    lvl = fget(form, 'price_level', np.nan)
    if "priceLevel" in df.columns:
        lvl_num = (np.nan if (lvl is None or (isinstance(lvl, float) and np.isnan(lvl))) else float(lvl))
        data["f_price_absdiff"] = (
            np.abs(df["priceLevel"].astype(float) - (0.0 if np.isnan(lvl_num) else lvl_num))
              .fillna(0.0).to_numpy(dtype=np.float32) / 3.0
        )

    # nombre d’options demandées
    req_opts = _extract_requested_options(form, df)
    data["f_req_count"] = np.full(len(df), float(len(req_opts) if req_opts else 0.0), dtype=np.float32)

    # cosinus sûr avec la description (dans [0,1])
    q = (fget(form, 'description', '') or '').strip()
    zf = None
    if q:
        zf = sent_model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    if "desc_embed" in df.columns:
        cos = (
            df["desc_embed"]
              .apply(lambda e: _cos01_safe(e, zf))
              .astype(np.float32)
              .to_numpy()
        )
    else:
        cos = np.zeros(len(df), dtype=np.float32)
    data["f_text_desc_cos"] = cos

    # quelques brutes utiles
    for c in ["rating", "priceLevel", "start_price", "latitude", "longitude"]:
        if c in df.columns:
            data[f"raw_{c}"] = df[c].astype(float).to_numpy(dtype=np.float32)

    X_df = pd.DataFrame(data, index=df.index)
    if "id_etab" in df.columns:
        X_df["id_etab"] = df["id_etab"].values

    gains = aggregate_gains(H, W_proxy)
    return X_df, gains
