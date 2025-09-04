import os
from pathlib import Path
import numpy as np
import pandas as pd

import benchmark_3 as e3

def _fake_catalog():
    df_etab = pd.DataFrame([
        {"id_etab": 1, "rating": 4.2, "priceLevel": "PRICE_LEVEL_MODERATE",
         "latitude": 47.39, "longitude": 0.69, "adresse": "20 Rue AAA, 37000 Tours",
         "editorialSummary_text": "Italien cosy", "start_price": 18.0},
        {"id_etab": 2, "rating": 3.6, "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
         "latitude": 47.40, "longitude": 0.68, "adresse": "5 Ave BBB, 37100 Tours",
         "editorialSummary_text": "Terrasse sympa", "start_price": 12.0},
        {"id_etab": 3, "rating": 4.8, "priceLevel": "PRICE_LEVEL_EXPENSIVE",
         "latitude": 47.41, "longitude": 0.67, "adresse": "10 Rue CCC, 37200 Tours",
         "editorialSummary_text": "Gastro", "start_price": 35.0},
    ])
    df_options = pd.DataFrame([
        {"id_etab": 1, "servesVegetarianFood": True,  "outdoorSeating": False, "restroom": True},
        {"id_etab": 2, "servesVegetarianFood": False, "outdoorSeating": True,  "restroom": True},
        {"id_etab": 3, "servesVegetarianFood": True,  "outdoorSeating": True,  "restroom": True},
    ])
    desc = [np.array([1,0,0], np.float32),
            np.array([0,1,0], np.float32),
            np.array([0,0,1], np.float32)]
    revs = [
        [np.array([0.2,0.8,0], np.float32)],
        [np.array([0.5,0.5,0], np.float32)],
        [np.array([0.9,0.1,0], np.float32)],
    ]
    df_embed = pd.DataFrame([
        {"id_etab": 1, "desc_embed": desc[0], "rev_embeds": revs[0]},
        {"id_etab": 2, "desc_embed": desc[1], "rev_embeds": revs[1]},
        {"id_etab": 3, "desc_embed": desc[2], "rev_embeds": revs[2]},
    ])
    df_open = pd.DataFrame([
        {"id_etab": 1, "open_day": 6, "open_hour": 19, "close_day": 6, "close_hour": 22},
        {"id_etab": 2, "open_day": 6, "open_hour": 12, "close_day": 6, "close_hour": 14},
        {"id_etab": 3, "open_day": 6, "open_hour": 12, "close_day": 6, "close_hour": 23},
    ])
    return {"etab": df_etab, "options": df_options, "etab_embedding": df_embed, "opening_period": df_open}

def fake_sentence_encoder(monkeypatch):
    class _FakeEncoder:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            if isinstance(texts, str):
                texts = [texts]
            out = []
            for t in texts:
                L = len(str(t))
                vec = np.array([L % 3 == 0, L % 3 == 1, L % 3 == 2], dtype=float)
                if vec.sum() == 0:
                    vec = np.array([1,0,0], dtype=float)
                if normalize_embeddings:
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / n
                out.append(vec.astype(np.float32))
            return np.vstack(out)
    monkeypatch.setattr(e3, "SENT_MODEL", _FakeEncoder(), raising=True)

def _forms_df():
    return pd.DataFrame([
        {"price_level": 2, "code_postal": "37000", "open": "ouvert_samedi_soir",
         "options": "servesVegetarianFood, outdoorSeating", "description": "italien calme terrasse"},
        {"price_level": 1, "code_postal": "37100", "open": "ouvert_samedi_midi",
         "options": "outdoorSeating", "description": "pas trop cher en terrasse"},
        {"price_level": 3, "code_postal": "37200", "open": "ouvert_samedi_soir",
         "options": "servesVegetarianFood", "description": "gastro special"},
    ])

def test_build_pointwise_and_pairwise_shapes_and_values(monkeypatch, tmp_path):
    # Patch data + encoder
    monkeypatch.setattr(e3.API.entrainement.Extract, "main", lambda: _fake_catalog())
    fake_sentence_encoder(monkeypatch)

    # Catalogue + préproc
    df = e3.load_and_prepare_catalog()
    preproc = e3.build_preproc_for_items(df)
    X_items = preproc.fit_transform(df)
    X_items = X_items.toarray().astype(np.float32) if hasattr(X_items, "toarray") else np.asarray(X_items, np.float32)

    n_items = len(df)
    d = X_items.shape[1]

    # 2 formulaires en "train", 1 en "test"
    forms = _forms_df()
    forms_tr = list(forms.iloc[:2].to_dict("records"))
    forms_te = list(forms.iloc[2:].to_dict("records"))

    # ---------- POINTWISE ----------
    Xtr, ytr, swtr, qtr = e3.build_pointwise(forms_tr, preproc, df, X_items, e3.SENT_MODEL)
    Xte, yte, swte, qte = e3.build_pointwise(forms_te, preproc, df, X_items, e3.SENT_MODEL)

    # tailles
    assert Xtr.shape == (len(forms_tr) * n_items, d + 2)
    assert Xte.shape == (len(forms_te) * n_items, d + 2)
    assert ytr.shape == (len(forms_tr) * n_items,)
    assert swtr.shape == ytr.shape
    assert qtr.shape == ytr.shape

    # types/plages
    assert Xtr.dtype == np.float32
    assert np.isfinite(Xtr).all()
    assert set(np.unique(ytr)).issubset({0, 1})
    assert np.isfinite(swtr).all() and (swtr > 0).all()  
    assert set(qtr) == set(range(len(forms_tr)))
    for qid in range(len(forms_tr)):
        assert (qtr == qid).sum() == n_items

    # Cohérence interne pour le 1er formulaire (bloc de n_items lignes)
    f0 = forms_tr[0]
    Zf_sp = preproc.transform(e3.form_to_row(f0, df))
    Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]
    T0 = e3.text_features01(df, f0, e3.SENT_MODEL, k=e3.PROXY_K)
    Xq_expected = e3.pair_features(Zf, X_items, T0)
    np.testing.assert_allclose(Xtr[:n_items], Xq_expected, atol=1e-6)

    # Les 2 dernières colonnes de Xtr doivent être les features texte (dans [0,1])
    np.testing.assert_allclose(Xtr[:n_items, -2:], T0, atol=1e-6)
    assert ((Xtr[:, -2:] >= 0) & (Xtr[:, -2:] <= 1)).all()

    # Labels y et poids sw correspondant aux gains proxy (recalculés)
    H_no_text = {
        'price':   e3.h_price_vector_simple(df, f0),
        'rating':  e3.h_rating_vector(df),
        'city':    e3.h_city_vector(df, f0),
        'options': e3.h_opts_vector(df, f0),
        'open':    e3.h_open_vector(df, f0, unknown_value=1.0),
    }
    text_proxy = e3.PROXY_W_REV * T0[:, 1] + (1.0 - e3.PROXY_W_REV) * T0[:, 0]
    gains0 = e3.aggregate_gains({**H_no_text, 'text': text_proxy}, e3.W_proxy)
    y0_expected = (gains0 >= e3.tau).astype(int)
    sw0_expected = np.abs(gains0 - e3.tau) + 1e-3
    np.testing.assert_array_equal(ytr[:n_items], y0_expected)
    np.testing.assert_allclose(swtr[:n_items], sw0_expected, atol=1e-9)

    # Sanity: pas tous 0 ni tous 1 (sur ce dataset jouet, ça doit varier)
    assert ytr.sum() > 0 and ytr.sum() < ytr.size

    # ---------- PAIRWISE ----------
    Xp, yp, wp = e3.build_pairwise(forms_tr, preproc, df, X_items, e3.SENT_MODEL, W=e3.W_proxy, top_m=10, bot_m=10)
    assert Xp.dtype == np.float32
    assert Xp.shape[1] == d + 2
    assert set(np.unique(yp)).issubset({1})  # que des positifs dans cette construction
    assert np.isfinite(wp).all()
    assert wp.min() >= 0.0 and wp.max() > 0.0  # il peut exister des 0 si ip==ineg, mais au moins un >0

    # Reconstitution exacte du 1er bloc de paires (pour le 1er formulaire)
    # même logique que dans build_pairwise
    order = np.argsort(gains0)
    pos_idx = order[::-1][:min(10, n_items)]
    neg_idx = order[:min(10, n_items)]

    Xq0 = Xq_expected
    Xp_expected, wp_expected = [], []
    for ip in pos_idx:
        for ineg in neg_idx:
            Xp_expected.append(Xq0[ip] - Xq0[ineg])
            wp_expected.append(float(gains0[ip] - gains0[ineg]))
    Xp_expected = np.vstack(Xp_expected).astype(np.float32)
    wp_expected = np.array(wp_expected, dtype=float)

    # Compare avec le début de Xp / wp (le 1er formulaire est traité en premier)
    n_first_block = len(Xp_expected)
    np.testing.assert_allclose(Xp[:n_first_block], Xp_expected, atol=1e-6)
    np.testing.assert_allclose(wp[:n_first_block], wp_expected, atol=1e-9)


def test_train_and_eval_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(e3.API.entrainement.Extract, "main", lambda: _fake_catalog())
    fake_sentence_encoder(monkeypatch)

    forms = pd.DataFrame([
        {"price_level": 2, "code_postal": "37000", "open": "ouvert_samedi_soir",
         "options": "servesVegetarianFood, outdoorSeating", "description": "italien calme terrasse"},
        {"price_level": 1, "code_postal": "37100", "open": "ouvert_samedi_midi",
         "options": "outdoorSeating", "description": "pas trop cher en terrasse"},
        {"price_level": 3, "code_postal": "37200", "open": "ouvert_samedi_soir",
         "options": "servesVegetarianFood", "description": "gastro special"},
    ])
    forms_csv = tmp_path / "forms.csv"
    forms.to_csv(forms_csv, index=False)

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        os.environ["FORMS_CSV"] = str(forms_csv)

        e3.main_entrainement()

        assert Path("artifacts/benchmark_summary.csv").exists()
        assert Path("artifacts/benchmark_per_query.csv").exists()
        assert Path("artifacts/preproc_items.joblib").exists()
        assert Path("artifacts/rank_model.joblib").exists()

        summary = pd.read_csv("artifacts/benchmark_summary.csv", index_col=0)
        for col in summary.columns:
            if col not in ["DCG@5_mean","DCG@5_lo95","DCG@5_hi95"]:
                vals = summary[col].values
                assert np.isfinite(vals).all()
                assert (vals >= -1.000001).all() and (vals <= 1.000001).all()
    finally:
        os.chdir(cwd)
        os.environ.pop("FORMS_CSV", None)
