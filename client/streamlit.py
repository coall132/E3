import os
import time
import json
import requests
import streamlit as st
from typing import Any, Dict, List, Optional

DEFAULT_API_BASE = os.getenv("API_BASE_URL")

st.set_page_config(page_title="Reco Restaurants — Client API", layout="wide")
st.title("Reco Restaurants — Client Streamlit (API externe)")

def _now_ts():
    return int(time.time())

def _normalize_base(url: str):
    return (url or "").strip().rstrip("/")

def _api_post(url: str, *, json_body: Optional[dict]=None, headers: Optional[dict]=None, params: Optional[dict]=None, timeout: int=30):
    r = requests.post(url, json=json_body, headers=headers, params=params, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        try:
            payload = r.json()
        except Exception:
            payload = {"detail": r.text}
        msg = payload.get("detail") or payload
        raise RuntimeError(f"HTTP {r.status_code} — {msg}") from e
    return r.json()

def _bearer_headers():
    tok = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {tok}"} if tok else {}

def _token_is_valid():
    exp = st.session_state.get("token_expires_at")
    if not exp:
        return False
    return _now_ts() < int(exp) - 10

def _pick_first(d, keys, default=None):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", [], {}):
            return v
    return default

def init_base_url():
    if "base_url" not in st.session_state:
        st.session_state["base_url"] = _normalize_base(DEFAULT_API_BASE)

with st.sidebar:
    st.header("Connexion API")
    base_url = st.text_input("Base URL de l'API", value=DEFAULT_API_BASE)
    st.session_state["base_url"] = _normalize_base(base_url)

    st.markdown("---")
    st.subheader("API Key")
    mode_have_key = st.toggle("J'ai déjà une API key", value=False)

    if mode_have_key:
        api_key_input = st.text_input("API Key (X-API-KEY)", type="password")
        if api_key_input:
            st.session_state["api_key"] = api_key_input.strip()
            st.success("API key enregistrée en session.")
    else:
        email = st.text_input("Email")
        username = st.text_input("Username (unique)")
        static_password = st.text_input("Mot de passe API (doit = API_STATIC_KEY côté serveur)", type="password")
        if st.button("Créer une API key", use_container_width=True, disabled=not (email and username and static_password)):
            try:
                url = f"{st.session_state['base_url']}/auth/api-keys"
                payload = {"email": email.strip(), "username": username.strip()}
                resp = _api_post(url, json_body=payload, params={"password": static_password})
                st.session_state["api_key"] = resp.get("api_key")
                st.session_state["api_key_id"] = resp.get("key_id")
                st.success("API key créée")
                st.code(st.session_state["api_key"], language="bash")
            except Exception as e:
                st.error(f"Échec création API key : {e}")

    st.markdown("---")
    st.subheader("Jeton d'accès")

    if st.button("Obtenir / Rafraîchir le token", use_container_width=True, disabled=not st.session_state.get("api_key")):
        try:
            url = f"{st.session_state['base_url']}/auth/token"
            resp = _api_post(url, headers={"X-API-KEY": st.session_state["api_key"]})
            st.session_state["access_token"] = resp.get("access_token")
            st.session_state["token_expires_at"] = resp.get("expires_at") 
            st.success("Token récupéré")
            st.caption(f"Expire à (epoch): {st.session_state['token_expires_at']}")
        except Exception as e:
            st.error(f"Échec token : {e}")

    if _token_is_valid():
        st.success("Token valide")
    else:
        st.warning("Pas de token valide. Récupérez-en un.")


st.subheader("Requête")
disabled_predict = not _token_is_valid()

with st.form("predict_form", clear_on_submit=False):
    c1, c2 = st.columns([1,1])
    with c1:
        price_level = st.selectbox("Gamme de prix", options=["(non précisé)", 1, 2, 3, 4], index=0)
        city = st.text_input("Ville", value="")
        open_str = st.text_input("Ouverture (texte libre)", value="", placeholder="ex: 'ouvert maintenant'")
    with c2:
        options_str = st.text_input("Options (séparées par des virgules)", value="", placeholder="ex: terrasse,wifi,goodforchildren")
        description = st.text_area("Description / préférences", height=100, placeholder="Ex: italien cosy, terrasse, budget moyen…")
    c3, c4, c5 = st.columns([1,1,2])
    with c3:
        k = st.slider("k (nb de résultats)", 1, 20, 5)
    with c4:
        use_ml = st.checkbox("use_ml", value=True)
    submitted = st.form_submit_button("Lancer /predict", disabled=disabled_predict)

if submitted:
    form = {
        "price_level": None if price_level == "(non précisé)" else price_level,
        "city": city or None,
        "open": open_str or "",
        "options": [o.strip() for o in options_str.replace(";", ",").split(",") if o.strip()] if options_str else [],
        "description": description or "",
    }
    try:
        url = f"{st.session_state['base_url']}/predict"
        params = {"k": k, "use_ml": str(use_ml).lower()}
        resp = _api_post(url, json_body=form, headers=_bearer_headers(), params=params)
        st.session_state["last_prediction"] = resp
        st.success("Prédiction OK")

        items_rich = resp.get("items_rich", [])
        rows = []
        for it in items_rich:
            d = it.get("details") or {}
            row = {
                "rank": it.get("rank"),
                "score": it.get("score"),
                "id_etab": it.get("etab_id"),
                "name": _pick_first(d, ["name", "nom", "title", "libelle"]),
                "city": _pick_first(d, ["city", "ville"]),
                "rating": _pick_first(d, ["rating", "note"]),
                "price_level": _pick_first(d, ["price_level", "prix"]),
                "options": _pick_first(d, ["options"]),
                "description": _pick_first(d, ["description", "desc"]),
            }
            rows.append(row)

        st.write(f"**prediction_id**: `{resp.get('prediction_id') or resp.get('id')}` | **k**: {resp.get('k')} | **model**: {resp.get('model_version', 'n/a')}")
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun champ attendu trouvé dans `items_rich`. Voir la réponse brute ci-dessous.")

        with st.expander("Réponse JSON brute"):
            st.json(resp)
    except Exception as e:
        st.error(f"Erreur /predict : {e}")

st.divider()

st.subheader("Envoyer un feedback")
pred_id = None
if st.session_state.get("last_prediction"):
    pred_id = st.session_state["last_prediction"].get("prediction_id") or st.session_state["last_prediction"].get("id")

c1, c2 = st.columns([1,3])
with c1:
    rating = st.selectbox("Note (0–5)", options=[0,1,2,3,4,5], index=5)
with c2:
    comment = st.text_input("Commentaire (optionnel)", value="")

disabled_fb = not (_token_is_valid() and pred_id)
if st.button("Envoyer /feedback", disabled=disabled_fb):
    try:
        url = f"{st.session_state['base_url']}/feedback"
        payload = {"prediction_id": str(pred_id), "rating": int(rating), "comment": comment or None}
        resp = _api_post(url, json_body=payload, headers=_bearer_headers())
        st.success("Feedback envoyé")
        with st.expander("Réponse brute"):
            st.json(resp)
    except Exception as e:
        st.error(f"Erreur /feedback : {e}")

st.caption("Ce client n'effectue **aucun calcul local** : il consomme uniquement vos endpoints FastAPI (auth/token/predict/feedback).")
