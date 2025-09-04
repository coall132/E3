# conftest.py
"""
Boot Postgres éphémère (testcontainers) AVANT l'import des tests,
override de API.database.engine / SessionLocal, création des tables,
puis import de API.main pour que les tests qui font `from API import main`
récupèrent le module déjà configuré.
"""
import os
import sys
import contextlib
import importlib
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer


# ---------- helpers ----------
def _engine_url_from_pg(pg: PostgresContainer) -> str:
    url = pg.get_connection_url()  # ex: postgresql://user:pass@127.0.0.1:xxxxx/db
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


# ---------- DÉMARRAGE TRES TÔT (avant collecte des tests) ----------
# Pytest importe conftest.py avant les fichiers de tests -> on peut démarrer ici.
_PG = PostgresContainer("postgres:16-alpine")
_PG.start()

# Variables d'env pour ton code (au cas où il s’y réfère)
os.environ.setdefault("DISABLE_WARMUP", "1")      # évite le warmup ML pendant tests
os.environ.setdefault("SKIP_RANK_MODEL", "1")
os.environ.setdefault("API_STATIC_KEY", "coall")
os.environ.setdefault("JWT_SECRET", "coall")
os.environ.setdefault("REDIS_URL", "memory://")   # limiter en mémoire, pas de Redis requis

os.environ["POSTGRES_USER"] = _PG.username
os.environ["POSTGRES_PASSWORD"] = _PG.password
os.environ["POSTGRES_DB"] = _PG.dbname
os.environ["POSTGRES_HOST"] = _PG.get_container_host_ip()
os.environ["POSTGRES_PORT"] = _PG.get_exposed_port(5432)

engine_url = _engine_url_from_pg(_PG)
os.environ["DATABASE_URL"] = engine_url
os.environ["SQLALCHEMY_DATABASE_URL"] = engine_url

# Override immédiat de l’engine / SessionLocal AVANT d’importer API.main
database = importlib.import_module("API.database")
with contextlib.suppress(Exception):
    database.engine.dispose()

database.engine = create_engine(engine_url, future=True)
database.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=database.engine)

# Schéma + tables
models = importlib.import_module("API.models")
models.ensure_ml_schema(database.engine)
models.Base.metadata.create_all(database.engine)

# Import **maintenant** l’app ; restera en cache pour les tests
main = importlib.import_module("API.main")
_APP = main.app

# ---------- hooks/fixtures ----------
def pytest_sessionfinish(session, exitstatus):
    # arrêt du container à la fin
    with contextlib.suppress(Exception):
        _PG.stop()


@pytest.fixture(scope="session")
def app():
    """Expose l'app si besoin dans certains tests."""
    return _APP


@pytest.fixture(scope="session")
def client_realdb(app):
    """Client FastAPI branché sur le Postgres testcontainers."""
    return TestClient(app)


@pytest.fixture
def db_session():
    """Session courte si certains tests en ont besoin directement."""
    s = database.SessionLocal()
    try:
        yield s
    finally:
        with contextlib.suppress(Exception):
            s.rollback()
        s.close()
