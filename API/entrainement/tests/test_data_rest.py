import numpy as np
import pandas as pd
import benchmark_3 as e3

def _fake_catalog():
    df_etab = pd.DataFrame([
        {
            "id_etab": 1,
            "rating": 4.2,
            "priceLevel": "PRICE_LEVEL_MODERATE",
            "latitude": 47.39,
            "longitude": 0.69,
            "adresse": "20 Rue Mirabeau, 37000 Tours",
            "editorialSummary_text": "Petit italien cosy",
            "start_price": 18.0,
        },
        {
            "id_etab": 2,
            "rating": np.nan,  
            "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
            "latitude": 47.40,
            "longitude": 0.68,
            "adresse": "5 Avenue Truc, 37100 Tours",
            "editorialSummary_text": "Terrasse sympa",
            "start_price": np.nan,  
        },
    ])
    df_options = pd.DataFrame([
        {"id_etab": 1, "servesVegetarianFood": True, "outdoorSeating": False, "restroom": None},
        {"id_etab": 2, "servesVegetarianFood": None, "outdoorSeating": True, "restroom": True},
    ])

    desc1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    desc2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    revs1 = [np.array([0.2, 0.8, 0.0], dtype=np.float32)]
    revs2 = [np.array([0.5, 0.5, 0.0], dtype=np.float32), np.array([0.3, 0.7, 0.0], dtype=np.float32)]
    df_embed = pd.DataFrame([
        {"id_etab": 1, "desc_embed": desc1, "rev_embeds": revs1},
        {"id_etab": 2, "desc_embed": desc2, "rev_embeds": revs2},
    ])

    df_open = pd.DataFrame([
        {"id_etab": 1, "open_day": 6, "open_hour": 19, "close_day": 6, "close_hour": 22},
        {"id_etab": 2, "open_day": 6, "open_hour": 12, "close_day": 6, "close_hour": 14},
    ])

    return {"etab": df_etab,"options": df_options,"etab_embedding": df_embed,"opening_period": df_open}

def test_data(monkeypatch):
    monkeypatch.setattr(e3.API.entrainement.Extract, "main", lambda: _fake_catalog())
    df = e3.load_and_prepare_catalog()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ["id_etab", "rating", "priceLevel", "code_postal", "desc_embed", "rev_embeds"]:
        assert col in df.columns

    assert all(isinstance(c, str) for c in df.columns)

    assert df["priceLevel"].dtype.kind in ("i", "f")  # numérique
    assert df["code_postal"].notna().all()

    de0 = df.loc[df.index[0], "desc_embed"]
    assert isinstance(de0, np.ndarray) and de0.ndim == 1

    rl1 = df.loc[df.index[1], "rev_embeds"]
    assert isinstance(rl1, list) and all(isinstance(v, np.ndarray) for v in rl1)

    ouvert_cols = [c for c in df.columns if c.startswith("ouvert_")]
    assert len(ouvert_cols) > 0

    assert df["rating"].notna().all()
    assert len(df) == 2


def test_preproc_and_fit_transform(monkeypatch):
    monkeypatch.setattr(e3.API.entrainement.Extract, "main", lambda: _fake_catalog())
    monkeypatch.setattr(e3.API.utils, "to_int_safe",
                        lambda X: pd.DataFrame(X).astype(int), raising=False)

    # 1) catalogue prêt
    df = e3.load_and_prepare_catalog()
    assert isinstance(df, pd.DataFrame) and not df.empty

    preproc = e3.build_preproc_for_items(df)
    # Doit contenir au moins num, bool, lev ; cp si 'code_postal' est là
    names = [t[0] for t in preproc.transformers]
    assert "num"  in names
    assert "bool" in names
    assert "lev"  in names
    if "code_postal" in df.columns:
        assert "cp" in names

    # 3) fit_transform sur le catalogue
    X_items = preproc.fit_transform(df)
    X_dense = X_items.toarray() if hasattr(X_items, "toarray") else np.asarray(X_items)
    assert X_dense.shape[0] == len(df)

    # --- vérifie le nombre de colonnes attendu ---
    BOOL_COLS = [
        'allowsDogs','delivery','goodForChildren','goodForGroups','goodForWatchingSports',
        'outdoorSeating','reservable','restroom','servesVegetarianFood','servesBrunch',
        'servesBreakfast','servesDinner','servesLunch'
    ]
    num_cols_present  = [c for c in ['rating','start_price'] if c in df.columns]
    bool_cols_present = [c for c in BOOL_COLS if c in df.columns]
    n_num  = len(num_cols_present)                # 2 ici
    n_bool = len(bool_cols_present)               # 3 ici 
    n_lev  = 1 if 'priceLevel' in df.columns else 0
    if 'code_postal' in df.columns:
        ohe_cp = preproc.named_transformers_['cp'].named_steps['onehot']
        n_cp   = len(ohe_cp.categories_[0])      # 2 
    else:
        n_cp = 0
    expected_d = n_num + n_bool + n_lev + n_cp
    assert X_dense.shape[1] == expected_d

    # --- bloc booléens : valeurs 0/1 uniquement ---
    start_bool = n_num
    bool_block = X_dense[:, start_bool:start_bool + n_bool]
    assert ((bool_block == 0) | (bool_block == 1)).all()

    # --- bloc code postal : one-hot correct et stable ---
    if n_cp > 0:
        start_cp = n_num + n_bool + n_lev
        cp_block = X_dense[:, start_cp:start_cp + n_cp]
        # chaque ligne correspond à exactement 1 CP connu
        rowsum = cp_block.sum(axis=1)
        assert np.allclose(rowsum, np.ones(len(df)))

        # transform() sur un formulaire avec CP connu vs inconnu
        form_known = {"price_level": 2, "code_postal": df["code_postal"].iloc[0]}
        z_known = preproc.transform(e3.form_to_row(form_known, df))
        z_known = z_known.toarray()[0] if hasattr(z_known, "toarray") else np.asarray(z_known).ravel()
        assert z_known.shape[0] == expected_d

        form_unknown = {"price_level": 2, "code_postal": "99999"}  # inconnu -> tout zéro sur le bloc CP
        z_unknown = preproc.transform(e3.form_to_row(form_unknown, df))
        z_unknown = z_unknown.toarray()[0] if hasattr(z_unknown, "toarray") else np.asarray(z_unknown).ravel()
        assert z_unknown.shape[0] == expected_d

        cp_known   = z_known[start_cp:start_cp + n_cp]
        cp_unknown = z_unknown[start_cp:start_cp + n_cp]
        assert np.isclose(cp_known.sum(), 1.0)
        assert np.isclose(cp_unknown.sum(), 0.0)

    # 4) Sanity check général : pas tout zéro
    assert (np.abs(X_dense).sum(axis=1) > 0).all()
