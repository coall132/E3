import os
from pathlib import Path
import numpy as np
import pandas as pd

import entrainement.benchmark_3 as e3

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
        {"id_etab": 1, "servesVegetarianFood": True, "outdoorSeating": False, "restroom": True},
        {"id_etab": 2, "servesVegetarianFood": False, "outdoorSeating": True, "restroom": True},
        {"id_etab": 3, "servesVegetarianFood": True, "outdoorSeating": True, "restroom": True},
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
    return {"etab": df_etab,"options": df_options,"etab_embedding": df_embed,"opening_period": df_open}

def fake_sentence_encoder(monkeypatch, dim=3):
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

def test_train_and_eval_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(e3.Extract, "main", lambda: _fake_catalog())
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
        assert Path("artifacts/linear_svc_pointwise.joblib").exists()

        summary = pd.read_csv("artifacts/benchmark_summary.csv", index_col=0)
        for col in summary.columns:
            vals = summary[col].values
            assert np.isfinite(vals).all()
            assert (vals >= -1e-6).all() and (vals <= 1.0 + 1e-6).all()
    finally:
        os.chdir(cwd)
        os.environ.pop("FORMS_CSV", None)