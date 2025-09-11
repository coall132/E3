import os, time, socket, subprocess, requests, signal
import pytest, uvicorn
from multiprocessing import Process
from contextlib import closing
import chromium

# ---------- Utils ----------

def _free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _wait_http_ok(url, timeout=None):
    timeout = int(os.getenv("E2E_STARTUP_TIMEOUT", "90")) if timeout is None else timeout
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=2)
            if 200 <= r.status_code < 400:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Timeout waiting for {url}")

# ---------- Fixtures serveurs ----------

@pytest.fixture(scope="function")   # <-- au lieu de session
def live_api(monkeypatch):
    external = os.getenv("E2E_API_BASE")
    if external:
        base_url = external.rstrip("/")
        _wait_http_ok(base_url + "/")
        return base_url

    from API.main import app as fastapi_app
    api_port = _free_port()
    base_url = f"http://127.0.0.1:{api_port}"

    # Config API pour tests
    monkeypatch.setenv("DISABLE_WARMUP", "1")
    monkeypatch.setenv("SKIP_RANK_MODEL", "1")
    monkeypatch.setenv("API_STATIC_KEY", "testpass")

    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=api_port, log_level="warning")
    server = uvicorn.Server(config)
    proc = Process(target=server.run, daemon=True)
    proc.start()

    try:
        _wait_http_ok(base_url + "/")
        yield base_url
    finally:
        proc.terminate()
        proc.join(timeout=5)

@pytest.fixture(scope="function")   # <-- idem
def live_streamlit(live_api):
    external = os.getenv("E2E_CLIENT_BASE")
    if external:
        st_url = external.rstrip("/")
        _wait_http_ok(st_url + "/_stcore/health")
        return st_url

    st_port = _free_port()
    st_url = f"http://127.0.0.1:{st_port}"

    env = os.environ.copy()
    env["API_BASE_URL"] = live_api
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        "streamlit", "run", "client/client_app.py",   # <-- adapte le chemin si besoin
        "--server.headless=true",
        f"--server.port={st_port}",
        "--browser.serverAddress=127.0.0.1",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        _wait_http_ok(st_url + "/_stcore/health")
        yield st_url
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ---------- Tests E2E avec Playwright ----------

@pytest.mark.e2e
def test_prediction(playwright, live_api, live_streamlit):
    browser = playwright.chromium.launch() 
    page = browser.new_page()
    page.goto(live_streamlit, wait_until="networkidle")

    # 1) Création API key
    page.get_by_label("Email").fill("alice@example.com")
    page.get_by_label("Username (unique)").fill("alice")
    page.get_by_label("Mot de passe API").fill(os.getenv("API_STATIC_KEY", "testpass"))
    page.get_by_role("button", name="Créer une API key").click()
    page.get_by_text("API key créée").wait_for(timeout=5000)

    # 2) Token
    page.get_by_role("button", name="Obtenir / Rafraîchir le token").click()
    page.get_by_text("Token récupéré").wait_for(timeout=5000)
    page.get_by_text("Token valide").wait_for(timeout=5000)

    # 3) /predict
    page.get_by_label("Gamme de prix").select_option(label="2")
    page.get_by_label("Ville").fill("Tours")
    page.get_by_label("Ouverture").fill("ouvert maintenant")
    page.get_by_label("Options").fill("terrasse,wifi")
    page.get_by_label("Description").fill("italien cosy, budget moyen")
    page.get_by_text("k (nb de résultats)").click()
    page.keyboard.press("ArrowRight")
    page.get_by_role("button", name="Lancer /predict").click()
    page.get_by_text("Prédiction OK").wait_for(timeout=15000)

    assert page.get_by_text("prediction_id").is_visible()

    # 4) /feedback
    page.get_by_role("button", name="Envoyer /feedback").click()
    page.get_by_text("Feedback envoyé").wait_for(timeout=5000)

    # 5) Déconnexion
    page.get_by_role("button", name="Se déconnecter (supprimer le token)").click()
    page.get_by_text("Token supprimé. Vous êtes déconnecté.").wait_for(timeout=5000)
    page.get_by_text("Pas de token valide").wait_for(timeout=5000)

    browser.close()