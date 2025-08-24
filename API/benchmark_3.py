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

model = SentenceTransformer('BAAI/bge-m3')

def fget(form, key, default=None):
    """Accès robuste à un champ de form (dict, pandas Series, namedtuple...)."""
    if isinstance(form, dict):
        return form.get(key, default)
    if isinstance(form, pd.Series):
        return form.get(key, default)
    return getattr(form, key, default)


def _iter_forms(forms):
    """Itère sur des formulaires venant d'un DataFrame ou d'une liste de dicts/tuples."""
    if isinstance(forms, pd.DataFrame):
        for row in forms.itertuples(index=False):
            yield row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())
    elif isinstance(forms, list):
        for f in forms:
            yield f
    else:
        yield forms

def embed(df, col, mode, sep="\n\n"):
    def avg_stack(arrs, eps=1e-9):
        if arrs is None or len(arrs) == 0:
            try:
                dim = model.get_sentence_embedding_dimension()
            except Exception:
                dim = len(model.encode([""], normalize_embeddings=True, show_progress_bar=False)[0])
            return np.zeros(dim, dtype=np.float32)
        M = np.vstack(arrs)
        m = M.mean(axis=0)
        n = np.linalg.norm(m)
        return (m / (n + eps)).astype(np.float32)

    def embedding(df, col):
        df[col] = df[col].fillna("").astype(str)
        emb_list = df[col].tolist()
        embeddings = model.encode(emb_list, normalize_embeddings=True, show_progress_bar=False)
        df[col] = list(embeddings)
        return df
    
    if mode == "list":
        def liste(x):
            if isinstance(x, list):
                L = [t for t in x if isinstance(t, str) and t.strip()]
            else:
                s = "" if x is None else str(x).strip()
                L = [s] if s else []
            if not L:
                return [] 
            E = model.encode(L, normalize_embeddings=True, show_progress_bar=False)
            return list(E)

        df = df.copy()
        df[col] = df[col].apply(liste)

    elif mode == "mean":
        def mean(x):
            if isinstance(x, list):
                L = [t for t in x if isinstance(t, str) and t.strip()]
            else:
                s = "" if x is None else str(x).strip()
                L = [s] if s else []
            if not L:
                return [] 
            E = model.encode(L, normalize_embeddings=True, show_progress_bar=False)
            return list(E)
        
        df = df.copy()
        df[col] = df[col].apply(mean)
        df[col] = df[col].apply(lambda x: avg_stack(x))
    else:
        df = embedding(df, col)
    return df

class EmbedWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, mode="mean", sep="\n\n"):
        self.mode = mode
        self.sep = sep
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        mats = []
        for c in Xdf.columns:
            df_c = Xdf[[c]].copy()
            df_c = embed(df_c, c, self.mode, self.sep)   
            vals = df_c[c].values
            M = np.vstack([v if isinstance(v, np.ndarray) else np.asarray(v, dtype=np.float32) for v in vals])
            mats.append(M.astype(np.float32))
        if not mats:
            return np.zeros((len(Xdf), 0), dtype=np.float32)
        return np.hstack(mats)


def _norm_txt(x):
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.casefold().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x).strip()
    return x

def city_to_postal_codes_exact(city: str,country: str = "FR"):
    if not city or not str(city).strip():
        return []

    if pgeocode is None:
        raise RuntimeError("pgeocode pas installé.")

    nomi = pgeocode.Nominatim(country.lower())
    loc = nomi.query_location(str(city).strip())
    if getattr(loc, "empty", True) or "postal_code" not in loc.columns:
        return []

    df = loc.copy()
    df["place_name"] = df.get("place_name", "").astype(str)
    df["postal_code"] = df.get("postal_code", "").astype(str)

    target = _norm_txt(city)
    df["norm_name"] = df["place_name"].apply(_norm_txt)

    mask = (df["norm_name"] == target) & (~df["norm_name"].str.contains("cedex"))

    df = df[mask]
    if df.empty:
        return []

    pcs = (df["postal_code"].dropna().astype(str).str.split(r"[,\s;]+").explode().str.strip().str.extract(r"^(\d{5})$")[0]
            .dropna().unique().tolist())

    pcs = sorted(pcs)
    return pcs

def h_price_vector_simple(df, form):
    lvl_f = fget(form, 'price_level', None)
    if lvl_f is None:
        return np.ones(len(df), dtype=float)
    diff = (df['priceLevel'].astype(float) - float(lvl_f)).abs()
    return (1.0 - (diff/3.0)).clip(0.0, 1.0).to_numpy(dtype=float)

def h_rating_vector(df, alpha=20.0):
    r = df['rating'].astype(float).fillna(2.5) if 'rating' in df.columns else pd.Series(2.5, index=df.index)
    mu = float(r.mean()) if r.notna().any() else 0.0

    if 'review_list' in df.columns:
        n = df['review_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).astype(float)
    else:
        n = pd.Series(1.0, index=df.index)  
    r_star = (n*r + alpha*mu) / (n + alpha)
    return np.clip(r_star/5.0, 0.0, 1.0).to_numpy(dtype=float)

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
    s = str(pcs).strip()
    return s

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
    q = (fget(form, 'description', '') or '').strip()
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
    score["price"]   = h_price_vector_simple(df, form)
    score["rating"]  = h_rating_vector(df, alpha=20.0)
    score["city"]    = h_city_vector(df, form)
    score["options"] = h_opts_vector(df, form)
    score["open"]    = h_open_vector(df,form, unknown_value=1.0, slot_prefix='ouvert_')
    score["text"] = score_text(df, form, model,w_rev=0.6, w_desc=0.4, k=3, missing_cos=0.0)
    return score


form = {
    'type': 'restaurant',         
    'price_level': 2,             
    'code_postal': ['37000','37100','37200'], 
    'open': 'ouvert_samedi_soir',          # colonne exacte dans df si dispo
    'options': ['servesVegetarianFood','outdoorSeating'],  # sous-ensemble de colonnes bool
    'description': 'italien calme et atypique, terrasse'   # texte libre
}

W_proxy = {'price':0.18,'rating':0.14,'options':0.14,'text':0.28,'city':0.04,'open':0.04}
W_eval  = {'price':0.12,'rating':0.18,'options':0.12,'text':0.22,'city':0.20,'open':0.16}
tau = 0.60

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
        row['code_postal'] = _first_code_postal(pcs)

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

def aggregate_gains(H, weights):
    """
    Combine les signaux H (dict de vecteurs par item) en un score unique par item,
    via une moyenne pondérée par 'weights'. Sortie: array (N_items,).
    """
    keys = list(H)  
    W = np.array([float(weights[k]) for k in keys], dtype=float)
    gains = np.zeros_like(np.asarray(H[keys[0]], dtype=float), dtype=float)
    for k, w in zip(keys, W):
        gains += w * np.asarray(H[k], dtype=float)
    return gains / W.sum()

def pair_features(Zf, H, X_items):
    diff = np.abs(X_items - Zf.reshape(1,-1))         # (N,D)
    cos  = np.asarray(H['text'], float)[:, None]      # (N,1)
    return np.hstack([diff, cos]).astype(np.float32)

def build_pointwise(forms_df, preproc, df, X_items, SENT_MODEL, tau=tau, W=W_proxy):
    """
    Génère un dataset pointwise:
      - X: une ligne par item (toutes les lignes de toutes les requêtes)
      - y: label binaire (gains >= tau)
      - sw: poids d'exemple = |gains - tau| + 1e-3
      - qid: id de la requête (pour group-by/éval)
    """
    X_list, y_list, sw_list, qid_list = [], [], [], []
    n_items = len(df)

    for qid, form in enumerate(_iter_forms(forms_df), start=0):
        # vecteur requête
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

        # signaux item→score
        H = {
            'price':  h_price_vector_simple(df, form),
            'rating': h_rating_vector(df),
            'city':   h_city_vector(df, form),
            'options':h_opts_vector(df, form),
            'open':   h_open_vector(df, form, unknown_value=1.0),
            'text':   score_text(df, form, SENT_MODEL),
        }

        gains = aggregate_gains(H, W)                 # (n_items,)
        y = (gains >= tau).astype(int)               # (n_items,)
        sw = np.abs(gains - tau) + 1e-3              # (n_items,)
        Xq = pair_features(Zf, H, X_items)           # (n_items, S + D + 1)

        X_list.append(Xq)
        y_list.append(y)
        sw_list.append(sw)
        qid_list.append(np.full(n_items, qid, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    sw = np.concatenate(sw_list)
    qid = np.concatenate(qid_list)
    return X, y, sw, qid

def build_pairwise(forms_df, preproc, df, X_items, SENT_MODEL,
                   tau=tau, W=W_proxy, top_m=10, bot_m=10):
    """
    Génère un dataset pairwise à la RankSVM:
      - on prend les 'top_m' meilleurs items et les 'bot_m' pires (proxy),
      - on crée des exemples (X_pos - X_neg) étiquetés 1,
      - poids = différence de gains.
    """
    Xp, yp, wp = [], [], []

    for form in _iter_forms(forms_df):
        # vecteur requête
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

        # signaux
        H = {
            'price':  h_price_vector_simple(df, form),
            'rating': h_rating_vector(df),
            'city':   h_city_vector(df, form),
            'options':h_opts_vector(df, form),
            'open':   h_open_vector(df, form, unknown_value=1.0),
            'text':   score_text(df, form, SENT_MODEL),
        }

        gains = aggregate_gains(H, W)                 # (n_items,)
        order = np.argsort(gains)
        pos_idx = order[::-1][:top_m]                 # meilleurs
        neg_idx = order[:bot_m]                       # pires

        Xq = pair_features(Zf, H, X_items)           # (n_items, S + D + 1)
        for ip in pos_idx:
            for ineg in neg_idx:
                Xp.append(Xq[ip] - Xq[ineg])
                yp.append(1)
                wp.append(float(gains[ip] - gains[ineg]))

    return np.vstack(Xp).astype(np.float32), np.array(yp), np.array(wp, dtype=float)

def eval_on_forms(forms_df, preproc, df, X_items, SENT_MODEL, models, k=5,
                  tau_q=0.85, use_top_m=None, jitter=1e-6):
    """
    Évalue Random, RuleProxy et les modèles sur un set de formulaires :
      - gold binaire défini par quantile (tau_q) ou top-M sur gains_eval (W_eval),
      - NDCG binaire (robuste aux gains plats),
      - RuleProxy classe avec W_proxy (pas un oracle),
      - moyenne P@k / NDCG@k sur toutes les requêtes.
    """
    out = {name: [] for name in (["Random", "RuleProxy"] + list(models.keys()))}

    for form in _iter_forms(forms_df):
        # vecteur requête
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]

        # signaux
        H = {
            'price':  h_price_vector_simple(df, form),
            'rating': h_rating_vector(df),
            'city':   h_city_vector(df, form),
            'options':h_opts_vector(df, form),
            'open':   h_open_vector(df, form, unknown_value=1.0),
            'text':   score_text(df, form, SENT_MODEL),
        }

        # GOLD/IDCG via W_eval (indépendant du proxy) + léger jitter
        gains_eval = aggregate_gains(H, W_eval).astype(float)
        if jitter:
            gains_eval = gains_eval + float(jitter) * np.random.randn(len(gains_eval))

        if use_top_m is not None:
            gold = set(np.argsort(gains_eval)[::-1][:int(use_top_m)])
        else:
            thr = np.quantile(gains_eval, float(tau_q))  # ex: 0.85 => top 15%
            gold = {i for i, g in enumerate(gains_eval) if g >= thr}

        # Baseline Random
        rand_pred = np.random.permutation(len(gains_eval))
        out["Random"].append((
            precision_at_k(rand_pred, gold, k),
            ndcg_binary_at_k(rand_pred, gold, k),
        ))

        # RuleProxy (classement selon le proxy d'entraînement)
        gains_proxy = aggregate_gains(H, W_proxy)
        pred_rule = np.argsort(gains_proxy)[::-1]
        out["RuleProxy"].append((
            precision_at_k(pred_rule, gold, k),
            ndcg_binary_at_k(pred_rule, gold, k),
        ))

        # Modèles
        Xq = pair_features(Zf, H, X_items)
        for name, m in models.items():
            if hasattr(m, "predict_proba"):
                scores = m.predict_proba(Xq)[:, 1]
            elif hasattr(m, "decision_function"):
                scores = m.decision_function(Xq)
            else:
                scores = m.predict(Xq).astype(float)

            pred = np.argsort(scores)[::-1]
            out[name].append((
                precision_at_k(pred, gold, k),
                ndcg_binary_at_k(pred, gold, k),
            ))

    # agrégation des métriques
    rows = {
        name: {
            f"Precision@{k}": float(np.mean([p for p, _ in vals])) if vals else 0.0,
            f"NDCG@{k}":      float(np.mean([n for _, n in vals])) if vals else 0.0,
        }
        for name, vals in out.items()
    }
    return pd.DataFrame(rows).T.sort_values(f"NDCG@{k}", ascending=False)

def precision_at_k(pred, gold, k):
    idx = np.asarray(pred, dtype=int)[:k]
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    hits = np.isin(idx, list(gold)).sum()
    return float(hits) / float(k) if k > 0 else 0.0

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]  # NumPy 2.0: asarray + dtype
    if r.size == 0:
        return 0.0
    return r[0] + (r[1:] / np.log2(np.arange(2, r.size + 1))).sum()

def ndcg_binary_at_k(pred, gold, k):
    """
    NDCG@k à pertinence binaire (1 si item ∈ gold, sinon 0).
    """
    n = len(pred)
    idx = np.asarray(pred, dtype=int)[:k]
    # vecteur de pertinence binaire
    rel = np.zeros(n, dtype=float)
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    if gold:
        rel[list(gold)] = 1.0

    dcg = dcg_at_k(rel[idx], k)

    # IDCG binaire = DCG d'une suite de '1' répétés min(k, |gold|) fois
    ideal_k = min(k, int(rel.sum()))
    if ideal_k == 0:
        return 0.0
    # gains = [1, 1, ..., 1] (ideal_k fois)
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
    """Average Precision @ k (binaire) sur une requête."""
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
    """Reciprocal Rank @ k."""
    if not isinstance(gold, (set, frozenset)):
        gold = set(gold)
    idx = np.asarray(pred, dtype=int)[:k]
    for i, it in enumerate(idx, start=1):
        if it in gold:
            return 1.0 / i
    return 0.0

def _bootstrap_ci(values, n_boot=300, alpha=0.05, rng=None):
    """IC bootstrap du mean (par défaut 95%). values: liste/array par requête."""
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

def forms_from_csv(path_csv: str, df_catalog: pd.DataFrame):
    df_forms = pd.read_csv(path_csv)
    forms = []
    for _, r in df_forms.iterrows():
        f = {}
        f['type'] = 'restaurant'
        # price_level
        if 'price_level' in df_forms.columns:
            val = r.get('price_level', None)
            try:
                if pd.notna(val) and str(val).strip() != '':
                    f['price_level'] = int(float(val))
            except Exception:
                pass
        # code_postal
        if 'code_postal' in df_forms.columns:
            cp = str(r.get('code_postal', '')).strip()
            if cp and cp.lower() != 'nan':
                f['code_postal'] = cp
        # open
        if 'open' in df_forms.columns:
            o = str(r.get('open', '')).strip()
            if o and o.lower() != 'nan':
                f['open'] = o
        # options
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
                    import re as _re
                    toks = [t.strip() for t in _re.split(r'[;,]', s) if t.strip()]
                    opts = toks
            if opts:
                f['options'] = opts
        # description
        if 'description' in df_forms.columns:
            d = r.get('description', '')
            if pd.notna(d):
                s = str(d).strip()
                if s and s.lower() != 'nan':
                    f['description'] = s
        forms.append(f)
    return forms

def eval_benchmark(forms_df, preproc, df, X_items, SENT_MODEL, models, k=5,
                   tau_q=0.85, use_top_m=None, jitter=1e-6, n_boot=300):
    """
    Benchmark 'léger' :
      - Precision@k, Recall@k, MAP@k, MRR@k, NDCG@k (binaire)
      - Moyenne par modèle + IC bootstrap (95%)
    Retour: (summary_df, per_query_df)
    """
    model_names = ["Random", "RuleProxy"] + list(models.keys())
    per_model = {name: [] for name in model_names}

    for form in _iter_forms(forms_df):
        # --- features requête & signaux item
        Zf_sp = preproc.transform(form_to_row(form, df))
        Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]
        H = {
            'price':  h_price_vector_simple(df, form),
            'rating': h_rating_vector(df),
            'city':   h_city_vector(df, form),
            'options':h_opts_vector(df, form),
            'open':   h_open_vector(df, form, unknown_value=1.0),
            'text':   score_text(df, form, SENT_MODEL),
        }

        # --- gold binaire via W_eval + léger jitter (tie-break)
        gains_eval = aggregate_gains(H, W_eval).astype(float)
        if jitter:
            gains_eval = gains_eval + float(jitter) * np.random.randn(len(gains_eval))
        if use_top_m is not None:
            gold = set(np.argsort(gains_eval)[::-1][:int(use_top_m)])
        else:
            thr = np.quantile(gains_eval, float(tau_q))  # ex: 0.85 => top 15 %
            gold = {i for i, g in enumerate(gains_eval) if g >= thr}

        # --- Random
        rand_pred = np.random.permutation(len(gains_eval))
        per_model["Random"].append({
            f"P@{k}": precision_at_k(rand_pred, gold, k),
            f"R@{k}": recall_at_k(rand_pred, gold, k),
            f"MAP@{k}": ap_at_k(rand_pred, gold, k),
            f"MRR@{k}": mrr_at_k(rand_pred, gold, k),
            f"NDCG@{k}": ndcg_binary_at_k(rand_pred, gold, k),
        })

        # --- RuleProxy (classement proxy d'entraînement)
        gains_proxy = aggregate_gains(H, W_proxy)
        pred_rule = np.argsort(gains_proxy)[::-1]
        per_model["RuleProxy"].append({
            f"P@{k}": precision_at_k(pred_rule, gold, k),
            f"R@{k}": recall_at_k(pred_rule, gold, k),
            f"MAP@{k}": ap_at_k(pred_rule, gold, k),
            f"MRR@{k}": mrr_at_k(pred_rule, gold, k),
            f"NDCG@{k}": ndcg_binary_at_k(pred_rule, gold, k),
        })

        # --- Modèles supervisés/non-supervisés
        Xq = pair_features(Zf, H, X_items)
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
                f"NDCG@{k}": ndcg_binary_at_k(pred, gold, k),
            })

    # --- agrégation + IC bootstrap
    rows, perq_rows = {}, []
    keep = [f"P@{k}", f"R@{k}", f"MAP@{k}", f"MRR@{k}", f"NDCG@{k}"]
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

import matplotlib.pyplot as plt

def plot_metric_bars(summary_df, metric, out_path):
    """
    Barres avec IC pour 'metric' (ex: 'NDCG@5').
    summary_df: index=modèle, colonnes: metric_mean/lo95/hi95
    """
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
    """
    Nuage P@k vs NDCG@k par modèle (moyenne).
    """
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


def make_unsup_view(Xq, keep_cos=True, D_diff=None):
    """
    Extrait une vue non-supervisée de Xq = [H_block | diff | cos].
    - On enlève le bloc H (dépend trop de la règle).
    - On garde diff (+ cos en option).
    - On scale et on compresse avec PCA (whiten) pour éviter l'écrasement par diff.
    """
    # Si tu connais D (dimension de Zf), D_diff = D, sinon on l’infère.
    # H_block = Xq[:,:S], diff = Xq[:,S:S+D], cos = Xq[:,-1]
    if D_diff is None:
        D_diff = Xq.shape[1] - 1  # suppose 1 colonne cos et zéro H (si tu as retiré H de pair_features)
    S = Xq.shape[1] - (D_diff + 1)
    H_block = Xq[:, :S] if S > 0 else None
    diff = Xq[:, S:S + D_diff]
    cos = Xq[:, -1:] if keep_cos else None

    parts = [diff]
    if keep_cos and cos is not None:
        parts.append(cos)
    Xu = np.hstack(parts).astype(np.float32)

    return Xu

class KMeansRanker:
    """
    Unsupervised KMeans sur une vue 'neutre' (diff + cos).
    Score = -distance au centroïde le plus proche (option: pondéré par taille de cluster).
    """
    def __init__(self, n_clusters=32, random_state=42, pca_dim=32, weight_by_size=True):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pca_dim = pca_dim
        self.weight_by_size = weight_by_size
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim, whiten=True, random_state=random_state)
        self.km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X, y=None):
        Xu = make_unsup_view(X)                 # extrait diff+cos
        Xs = self.scaler.fit_transform(Xu)
        Xp = self.pca.fit_transform(Xs)
        self.km.fit(Xp)
        # tailles de clusters (pour pondération optionnelle)
        labels = self.km.predict(Xp)
        self.cluster_sizes_ = np.bincount(labels, minlength=self.n_clusters) + 1e-9
        return self

    def decision_function(self, X):
        Xu = make_unsup_view(X)
        Xs = self.scaler.transform(Xu)
        Xp = self.pca.transform(Xs)
        d = self.km.transform(Xp)                       # (n_samples, n_clusters)
        mins = d.min(axis=1)
        if self.weight_by_size:
            nearest = d.argmin(axis=1)
            size = self.cluster_sizes_[nearest]
            return -(mins / np.log1p(size))             # plus grand = mieux
        return -mins

class AutoencoderRanker:
    """
    Unsupervised autoencoder sur diff+cos.
    mode='typical': score = -MSE (proximité du manifold global)
    mode='novel'  : score = +MSE (items atypiques mieux scorés)
    """
    def __init__(self, hidden=64, random_state=42, pca_dim=32, mode='typical', max_iter=300, alpha=1e-5):
        assert mode in ('typical','novel')
        self.hidden = hidden
        self.random_state = random_state
        self.pca_dim = pca_dim
        self.mode = mode
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim, whiten=True, random_state=random_state)
        self.ae = MLPRegressor(
            hidden_layer_sizes=(hidden,),
            activation='relu', solver='adam',
            max_iter=max_iter, random_state=random_state,
            alpha=alpha, verbose=False
        )

    def fit(self, X, y=None):
        Xu = make_unsup_view(X)
        Xs = self.scaler.fit_transform(Xu)
        Xp = self.pca.fit_transform(Xs)
        self.ae.fit(Xp, Xp)      # reconstruction
        return self

    def decision_function(self, X):
        Xu = make_unsup_view(X)
        Xs = self.scaler.transform(Xu)
        Xp = self.pca.transform(Xs)
        Xh = self.ae.predict(Xp)
        err = np.mean((Xp - Xh)**2, axis=1)
        return -err if self.mode == 'typical' else err
    