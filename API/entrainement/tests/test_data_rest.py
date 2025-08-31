import numpy as np
import pandas as pd
import entrainement.benchmark_3 as e3

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
    monkeypatch.setattr(e3.Extract, "main", lambda: _fake_catalog())
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