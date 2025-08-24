from typing import Any, List
import json
import numpy as np
import pandas as pd

def _parse_vec(v: Any):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=float)
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return np.asarray(json.loads(v), dtype=float)
        except Exception:
            return None
    return None

def _parse_mat(m: Any):
    if m is None:
        return []
    if isinstance(m, (bytes, bytearray)):
        m = m.decode("utf-8", errors="ignore")
    if isinstance(m, str):
        m = m.strip()
        if not m:
            return []
        try:
            m = json.loads(m)
        except Exception:
            return []
    if isinstance(m, (list, tuple)):
        out = []
        for row in m:
            try:
                out.append(np.asarray(row, dtype=float))
            except Exception:
                pass
        return out
    return []

def determine_price_level(row):
    if pd.notna(row['start_price']):
        price = row['start_price']
        if price < 15:
            return 1
        elif price > 15:
            return 2
        elif price > 20:
            return 3
    else :
        return np.nan
    
def calculer_profil_ouverture(row, df_horaires, jours, creneaux):
    etab_id = row['id_etab']  
    profil = {'id_etab': etab_id}
    
    for j in jours.values():
        for c in creneaux.keys():
            profil[f"ouvert_{j}_{c}"] = 0
    
    horaires_etab = df_horaires[df_horaires['id_etab'] == etab_id]
    
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