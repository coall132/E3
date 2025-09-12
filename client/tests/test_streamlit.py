import os, time, socket, subprocess, requests, signal, time
import pytest, uvicorn
from multiprocessing import Process
from contextlib import closing
import chromium
import sys
from pathlib import Path
from playwright.sync_api import expect

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
    from API import models
    from API.database import engine, get_db, SessionLocal
    models.ensure_ml_schema(engine)
    models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        etablissements = [
            models.Etablissement(
                id_etab=101, nom='Resto de Test A (Terrasse & Livraison)', adresse='1 Rue du Test, 37000 Tours',
                internationalPhoneNumber='+33 1 23 45 67 89', websiteUri='http://resto-a.test',
                rating=4.5, priceLevel='PRICE_LEVEL_MODERATE', start_price=15.0, end_price=30.0,
                latitude=47.38, longitude=0.68, editorialSummary_text='Un bon resto italien.'
            ),
            models.Etablissement(
                id_etab=102, nom='Bistrot Fictif B', adresse='2 Avenue de la Fiction, 37200 Tours',
                rating=4.0, priceLevel='PRICE_LEVEL_EXPENSIVE'
            ),
        ]

        options = [
            models.Options(
                id_etab=101, delivery=True, outdoorSeating=True,
                reservable=True, restroom=True, servesDinner=True, servesLunch=True
            ),
            models.Options(
                id_etab=102, delivery=False, outdoorSeating=False,
                reservable=True, restroom=True, servesDinner=True, servesLunch=True
            ),
        ]

        horaires = [
            models.OpeningPeriod(id_etab=101, open_day=1, open_hour=12, open_minute=0, close_day=1, close_hour=14, close_minute=0), # Lundi 12:00 - 14:00
            models.OpeningPeriod(id_etab=101, open_day=1, open_hour=19, open_minute=0, close_day=1, close_hour=22, close_minute=0), # Lundi 19:00 - 22:00
        ]

        objets_de_test = etablissements + options + horaires
        db.add_all(objets_de_test)
        db.commit()
        print(f"\n--- {len(objets_de_test)} objets de test insérés en BDD ---")

    finally:
        db.close()

    external = os.getenv("E2E_API_BASE")
    API_STATIC_KEY = os.getenv("API_STATIC_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    JWT_SECRET = os.getenv("JWT_SECRET")
    E2E = os.getenv("E2E","1")=="1"
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
    monkeypatch.setenv("API_STATIC_KEY", API_STATIC_KEY)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("E2E", E2E)

    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=api_port, log_level="info")
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
    client_root = Path(__file__).parent.parent 
    streamlit_script_path = client_root / "streamlit.py"

    cmd = [
        "streamlit", "run",  str(streamlit_script_path),
        "--server.headless=true",
        f"--server.port={st_port}",
        "--browser.serverAddress=127.0.0.1",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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
    results_dir = Path(__file__).parent / "test-results"
    results_dir.mkdir(exist_ok=True)
    browser = playwright.chromium.launch() 
    page = browser.new_page()
    page.goto(live_streamlit, wait_until="networkidle")

    timestamp = int(time.time())
    # 1) Création API key
    page.get_by_label("Email").fill(f"alice+{timestamp}@example.com")
    page.get_by_label("Username (unique)").fill(f"alice-{timestamp}")
    page.get_by_label("Mot de passe API").fill(os.getenv("API_STATIC_KEY"))
    page.get_by_role("button", name="Créer une API key").click()
    try:
        page.get_by_text("API key créée").wait_for(timeout=30000)
    except Exception as e:
        screenshot_path = results_dir / "echec-creation-cle.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        raise e

    # 2) Token
    page.get_by_role("button", name="Obtenir / Rafraîchir le token").click()
    page.get_by_text("Token récupéré").wait_for(timeout=15000)
    page.get_by_text("Token valide").wait_for(timeout=15000)

    # 3) /predict
    page.get_by_label("Gamme de prix").click()
    page.get_by_role("option", name="2").click()
    page.get_by_label("Ville").fill("37000")
    page.get_by_label("Ouverture").fill("ouvert maintenant")
    page.get_by_label("Options disponibles").click()
    page.get_by_text("Terrasse").click()
    page.get_by_text("Livraison").click()
    page.get_by_label("Description").click()
    page.get_by_label("Description").fill("italien cosy, budget moyen")
    page.get_by_text("k (nb de résultats)").click()
    page.keyboard.press("ArrowRight")
    page.get_by_role("button", name="Lancer /predict").click()

    expect(page.get_by_text("Appel /predict…")).not_to_be_visible(timeout=60000)
    expect(page.locator('[data-testid="stDataFrame"]')).to_be_visible()

    # 4) /feedback
    page.get_by_role("button", name="Envoyer /feedback").click()
    page.get_by_label("Note (0–5)").click() 
    page.get_by_role("option", name="4").click()
    page.get_by_text("Feedback envoyé").wait_for(timeout=15000)

    # 5) Déconnexion
    page.get_by_role("button", name="Se déconnecter (supprimer le token)").click()
    page.get_by_text("Token supprimé. Vous êtes déconnecté.").wait_for(timeout=15000)
    page.get_by_text("Pas de token valide").wait_for(timeout=15000)

    browser.close()