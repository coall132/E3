# tests/test_case3_inference.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import benchmark_3 as e3

def fake_catalog():
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
    return {"etab": df_etab,"options": df_options,"etab_embedding": df_embed,"opening_period": df_open}



def infer_scores_for_form(df, form, preproc, X_items, model):
    # 1) Embedding du formulaire
    Zf_sp = preproc.transform(e3.form_to_row(form, df))
    Zf = Zf_sp.toarray()[0] if hasattr(Zf_sp, "toarray") else np.asarray(Zf_sp)[0]
    # 2) Features texte (N,2) attendues par pair_features
    T_feat = e3.text_features01(df, form, e3.SENT_MODEL, k=e3.PROXY_K)
    # 3) Features finales pour le modèle
    Xq = e3.pair_features(Zf, X_items, T_feat)
    # 4) Scores
    if hasattr(model, "decision_function"):
        scores = model.decision_function(Xq)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(Xq)[:, 1]
    else:
        scores = model.predict(Xq).astype(float)
    ids = df["id_etab"].values
    return np.asarray(scores, float), ids

def test_inference_end_to_end(tmp_path, monkeypatch):
    # >>> patch correct de la source de données
    monkeypatch.setattr(e3.API.entrainement.Extract, "main", lambda: fake_catalog())

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

        # entraîne et produit artifacts/rank_model.joblib
        e3.main_entrainement()

        model_path = Path("artifacts/rank_model.joblib")
        assert model_path.exists()

        df = e3.load_and_prepare_catalog()
        preproc = e3.build_preproc_for_items(df)
        X_items = preproc.fit_transform(df)
        X_items = X_items.toarray().astype(np.float32) if hasattr(X_items, "toarray") else np.asarray(X_items, np.float32)

        model = joblib.load(model_path)

        # Cas 1 : len % 3 == 0 -> vecteur [1,0,0] -> item 1 favorisé
        form1 = {"description": "x" * 6}
        scores, ids = infer_scores_for_form(df, form1, preproc, X_items, model)
        order = np.argsort(scores)[::-1]
        assert len(scores) == len(df) == len(ids)
        assert np.all(scores[order[:-1]] >= scores[order[1:]])
        assert np.unique(scores).size > 1

        form2 = {"description": "xxxxxxx"}  # 7
        scores2, ids2 = infer_scores_for_form(df, form2, preproc, X_items, model)
        order2 = np.argsort(scores2)[::-1]
        assert int(ids[order[0]]) != int(ids2[order2[0]])


    finally:
        os.chdir(cwd)
        os.environ.pop("FORMS_CSV", None)
