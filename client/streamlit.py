import os
import time
import json
import requests
import streamlit as st
from typing import Any, Dict, List, Optional

DEFAULT_API_BASE = os.getenv("API_BASE_URL")

st.set_page_config(page_title="Reco Restaurants — Client API", layout="wide")
st.title("Reco Restaurants — Client Streamlit (API externe)")
DEFAULT_HTTP_TIMEOUT = int(os.getenv("API_HTTP_TIMEOUT", "360"))

DEFAULT_OPTIONS = [
    "allowsDogs", "delivery", "goodForChildren", "goodForGroups",
    "goodForWatchingSports", "outdoorSeating", "reservable", "restroom",
    "servesVegetarianFood", "servesBrunch", "servesBreakfast", "servesDinner", "servesLunch",
]

OPTION_LABELS = {
    "allowsDogs": "Animaux acceptés",
    "delivery": "Livraison",
    "goodForChildren": "Adapté aux enfants",
    "goodForGroups": "Groupes bienvenus",
    "goodForWatchingSports": "Diffusion d’événements sportifs",
    "outdoorSeating": "Terrasse",
    "reservable": "Réservable",
    "restroom": "Toilettes",
    "servesVegetarianFood": "Options végétariennes",
    "servesBrunch": "Brunch",
    "servesBreakfast": "Petit-déjeuner",
    "servesDinner": "Dîner",
    "servesLunch": "Déjeuner",
}
OPTION_KEYS = {
    "allowsDogs": "Chiens acceptés",
    "delivery": "Livraison",
    "goodForChildren": "Enfants bienvenus",
    "goodForGroups": "Groupes bienvenus",
    "goodForWatchingSports": "TV / sports",
    "outdoorSeating": "Terrasse",
    "reservable": "Réservable",
    "restroom": "Toilettes",
    "servesVegetarianFood": "Végétarien",
    "servesBrunch": "Brunch",
    "servesBreakfast": "Petit-déj",
    "servesDinner": "Dîner",
    "servesLunch": "Déjeuner",
}
def get_available_options():
    """
    Essaie de récupérer la liste d'options côté API (GET /meta/options).
    Si indisponible, on retourne la liste locale DEFAULT_OPTIONS.
    La valeur retournée DOIT être la clé attendue par le backend.
    """
    try:
        url = f"{_normalize_base(st.session_state.get('base_url') or DEFAULT_API_BASE)}/meta/options"
        r = requests.get(url, timeout=10)
        if r.ok:
            data = r.json()
            opts = data.get("options")
            if isinstance(opts, list) and all(isinstance(x, str) for x in opts):
                return opts
    except Exception:
        pass
    return DEFAULT_OPTIONS

def _to_bool(v):
    if isinstance(v, bool): 
        return v
    if v is None: 
        return False
    return str(v).strip().lower() in {"1","true","True"}

def _extract_options(details: dict) -> list[str]:
    """
    Cherche les booléens d'option soit à plat dans details,
    soit dans details['options'] si c'est un dict.
    """
    srcs = [details]
    if isinstance(details.get("options"), dict):
        srcs.append(details["options"])

    res = []
    for key, label in OPTION_KEYS.items():
        val = None
        for src in srcs:
            if key in src:
                val = src[key]
                break
        if _to_bool(val):
            res.append(label)
    return res

def _api_post(url: str, *, json_body=None, headers=None, params=None, timeout: int=DEFAULT_HTTP_TIMEOUT):
    r = requests.post(url, json=json_body, headers=headers, params=params, timeout=(5, timeout))
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
    http_timeout = st.number_input("Timeout HTTP (s)", min_value=10, max_value=600, value=120, step=5)
    st.session_state["http_timeout"] = int(http_timeout)
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

    if st.button("Se déconnecter (supprimer le token)", use_container_width=True, disabled=not st.session_state.get("access_token")):
        st.session_state.pop("access_token", None)
        st.session_state.pop("token_expires_at", None)
        st.success("Token supprimé. Vous êtes déconnecté.")
        st.rerun()


st.subheader("Requête")
disabled_predict = not _token_is_valid()

with st.form("predict_form", clear_on_submit=False):
    c1, c2 = st.columns([1,1])
    with c1:
        price_level = st.selectbox("Gamme de prix", options=["(non précisé)", 1, 2, 3, 4], index=0)
        city = st.text_input("Ville", value="")
        open_str = st.text_input("Ouverture (texte libre)", value="", placeholder="ex: 'ouvert maintenant'")
    with c2:
        all_opts = get_available_options()
        multi_opts = st.toggle("Sélection multiple d'options", value=True, key="opts_multi")

        if multi_opts:
            selected_opts = st.multiselect(
                "Options disponibles",
                options=all_opts,
                default=[],
                format_func=lambda x: OPTION_LABELS.get(x, x),  # affiche un label lisible
                help="Choisissez une ou plusieurs options"
            )
        else:
            selected_single = st.selectbox(
                "Option disponible",
                options=["(aucune)"] + all_opts,
                format_func=lambda x: OPTION_LABELS.get(x, x) if x != "(aucune)" else x,
                help="Sélection simple"
            )
            selected_opts = [] if selected_single == "(aucune)" else [selected_single]
        description = st.text_area("Description / préférences", height=100, placeholder="Ex: italien cosy, terrasse, budget moyen…")
    c3, c4, c5 = st.columns([1,1,2])
    with c3:
        k = st.slider("k (nb de résultats)", 1, 20, 5)
    with c4:
        use_ml = st.checkbox("use_ml", value=True)
    submitted = st.form_submit_button("Lancer /predict", disabled=disabled_predict)

if submitted:
    form = {
        "price_level": 2 if price_level == "(non précisé)" else price_level,
        "city": city or 37000,
        "open": open_str or "",
        "options": selected_opts,
        "description": description or "",
    }
    try:
        url = f"{st.session_state['base_url']}/predict"
        params = {"k": k, "use_ml": str(use_ml).lower()}
        to = st.session_state.get("http_timeout", DEFAULT_HTTP_TIMEOUT)
        with st.spinner(f"Appel /predict… (timeout {to}s)"):
            resp = _api_post(url, json_body=form, headers=_bearer_headers(), params=params, timeout=to)
        st.session_state["last_prediction"] = resp
        st.success("Prédiction OK")

        items_rich = resp.get("items_rich", [])
        rows = []
        for it in items_rich:
            d = it.get("details") or {}
            opts_list = _extract_options(d)
            row = {
                "rank": it.get("rank"),
                "score": it.get("score"),
                "id_etab": it.get("etab_id"),
                "name": _pick_first(d, ["name", "nom", "title", "libelle"]),
                "city": _pick_first(d, ['adresse',"code_postal", "cp"]),
                "rating": _pick_first(d, ["rating", "note"]),
                "price_level": _pick_first(d, ["priceLevel"]),
                "options": opts_list,
                "description": _pick_first(d, ["description", "desc"]),
            }
            rows.append(row)

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun champ attendu trouvé dans `items_rich`.")
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
        to = st.session_state.get("http_timeout", DEFAULT_HTTP_TIMEOUT)
        with st.spinner(f"Appel /feedback… (timeout {to}s)"):
            resp = _api_post(url, json_body=payload, headers=_bearer_headers(), timeout=to)
        st.success("Feedback envoyé")
    except Exception as e:
        st.error(f"Erreur /feedback : {e}")

