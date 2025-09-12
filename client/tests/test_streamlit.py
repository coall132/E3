import os, time, socket, subprocess, requests, signal
import pytest, uvicorn
from multiprocessing import Process
from contextlib import closing
import chromium
import sys

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
    # On ajoute une variable pour stocker les logs du process en cas de timeout
    proc_logs_for_timeout = None
    if "FAIL_FAST_WITH_LOGS" in globals():
        proc_logs_for_timeout = globals()["FAIL_FAST_WITH_LOGS"]

    while time.time() - t0 < timeout:
        # Si le process a crashé, on arrête tout de suite
        if proc_logs_for_timeout and proc_logs_for_timeout.poll() is not None:
             # On attend un peu pour que tous les logs soient bien récupérés
            time.sleep(1)
            stdout, stderr = proc_logs_for_timeout.communicate()
            print("--- Streamlit process crashed ---", file=sys.stderr)
            print(f"Return code: {proc_logs_for_timeout.returncode}", file=sys.stderr)
            if stdout:
                print("--- STDOUT ---", file=sys.stderr)
                print(stdout.decode(errors="ignore"), file=sys.stderr)
            if stderr:
                print("--- STDERR ---", file=sys.stderr)
                print(stderr.decode(errors="ignore"), file=sys.stderr)
            raise RuntimeError("Streamlit process failed to start.")

        try:
            r = requests.get(url, timeout=2)
            if 200 <= r.status_code < 400:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Timeout waiting for {url}")


# ---------- Fixtures serveurs ----------

@pytest.fixture(scope="function")
def live_api(monkeypatch):
    external = os.getenv("E2E_API_BASE")
    if external:
        base_url = external.rstrip("/")
        _wait_http_ok(base_url + "/")
        yield base_url
        return

    # 2) Mode local: spawn uvicorn
    from API.main import app as fastapi_app
    api_port = _free_port()
    base_url = f"http://127.0.0.1:{api_port}"

    monkeypatch.setenv("DISABLE_WARMUP", "1")
    monkeypatch.setenv("SKIP_RANK_MODEL", "1")
    monkeypatch.setenv("API_STATIC_KEY", "testpass")
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:////tmp/test.db")

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


@pytest.fixture(scope="function")
def live_streamlit(monkeypatch, live_api):
    external = os.getenv("E2E_CLIENT_BASE")
    if external:
        st_url = external.rstrip("/")
        _wait_http_ok(st_url + "/_stcore/health")
        yield st_url
        return

    st_port = _free_port()
    st_url = f"http://127.0.0.1:{st_port}"

    env = os.environ.copy()
    env["API_BASE_URL"] = live_api
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        "streamlit", "run", "streamlit.py",
        "--server.headless=true",
        f"--server.port={st_port}",
        "--browser.serverAddress=127.0.0.1",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # On stocke le process dans une variable globale pour que _wait_http_ok puisse le voir
    global FAIL_FAST_WITH_LOGS
    FAIL_FAST_WITH_LOGS = proc
    
    try:
        _wait_http_ok(st_url + "/_stcore/health")
        yield st_url
    finally:
        # On nettoie la variable globale
        if "FAIL_FAST_WITH_LOGS" in globals():
            del globals()["FAIL_FAST_WITH_LOGS"]

        proc.send_signal(signal.SIGINT)
        try:
            # On récupère les logs même si le process a bien fonctionné, pour le débogage
            stdout, stderr = proc.communicate(timeout=5)
            if stdout:
                print("\n--- Streamlit STDOUT ---")
                print(stdout.decode(errors="ignore"))
            if stderr:
                print("\n--- Streamlit STDERR ---")
                print(stderr.decode(errors="ignore"))
        except subprocess.TimeoutExpired:
            proc.kill()
            print("\n--- Streamlit process killed after timeout ---")

# ---------- Tests E2E avec Playwright ----------

@pytest.mark.e2e
def test_prediction(playwright, live_api, live_streamlit):
    browser = playwright.chromium.launch() 
    page = browser.new_page()
    page.goto(live_streamlit, wait_until="networkidle")

    # 1) Création API key
    page.get_by_label("Email").fill("alice@example.com")
    page.get_by_label("Username (unique)").fill("alice")
    page.get_by_label("Mot de passe API").fill(os.getenv("API_STATIC_KEY"))
    page.get_by_role("button", name="Créer une API key").click()
    page.get_by_text("API key créée").wait_for(timeout=30000)

    # 2) Token
    page.get_by_role("button", name="Obtenir / Rafraîchir le token").click()
    page.get_by_text("Token récupéré").wait_for(timeout=15000)
    page.get_by_text("Token valide").wait_for(timeout=15000)

    # 3) /predict
    page.get_by_label("Gamme de prix").select_option(label="2")
    page.get_by_label("Ville").fill("Tours")
    page.get_by_label("Ouverture").fill("ouvert maintenant")
    page.get_by_label("Options").fill("terrasse,wifi")
    page.get_by_label("Description").fill("italien cosy, budget moyen")
    page.get_by_text("k (nb de résultats)").click()
    page.keyboard.press("ArrowRight")
    page.get_by_role("button", name="Lancer /predict").click()
    page.get_by_text("Prédiction OK").wait_for(timeout=30000)

    assert page.get_by_text("prediction_id").is_visible()

    # 4) /feedback
    page.get_by_role("button", name="Envoyer /feedback").click()
    page.get_by_text("Feedback envoyé").wait_for(timeout=15000)

    # 5) Déconnexion
    page.get_by_role("button", name="Se déconnecter (supprimer le token)").click()
    page.get_by_text("Token supprimé. Vous êtes déconnecté.").wait_for(timeout=15000)
    page.get_by_text("Pas de token valide").wait_for(timeout=15000)

    browser.close()