# conftest.py
"""
Boot Postgres éphémère (testcontainers) AVANT l'import des tests,
override de API.database.engine / SessionLocal, création des tables,
puis import de API.main pour que les tests qui font `from API import main`
récupèrent le module déjà configuré.
"""

import os
import contextlib
import importlib

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

import numpy as np
import pandas as pd

def _ensure_minimal_app_state_for_api(app):
    class _FakeEncoder:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            if isinstance(texts, str):
                texts = [texts]
            out = []
            for t in texts:
                L = len(str(t))
                vec = np.array([L % 3 == 0, L % 3 == 1, L % 3 == 2], dtype=float)
                if vec.sum() == 0:
                    vec = np.array([1,0,0], dtype=float)
                if normalize_embeddings:
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / n
                out.append(vec.astype(np.float32))
            return np.vstack(out)

    if not hasattr(app.state, "SENT_MODEL"):
        app.state.SENT_MODEL = _FakeEncoder()

    if not hasattr(app.state, "DF_CATALOG"):
        df = pd.DataFrame({
            "id_etab": [1, 2, 3],
            "rating": [4.0, 3.5, 4.5],
            "priceLevel": [1, 2, 3],
            "latitude": [47.39, 47.39, 47.39],
            "longitude": [0.68, 0.68, 0.68],
            "editorialSummary_text": ["a", "b", "c"],
            "start_price": [10, 12, 30],
            "code_postal": ["37000", "37100", "37200"],
            "delivery": [True, False, True],
            "servesVegetarianFood": [True, True, False],
            # embeddings 3D alignés sur l’encoder dummy
            "desc_embed": [np.array([1,0,0], dtype=np.float32),
                           np.array([0,1,0], dtype=np.float32),
                           np.array([0,0,1], dtype=np.float32)],
            "rev_embeds": [None, None, None],
            # une colonne "open" utilisée par les tests
            "ouvert_lundi_midi": [1, 0, 1],
        })
        app.state.DF_CATALOG = df

def _engine_url_from_pg(pg: PostgresContainer) -> str:
    """
    Construit une URL SQLAlchemy psycopg2 à partir du container.
    testcontainers retourne typiquement 'postgresql://...'
    """
    url = pg.get_connection_url() 
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


_PG = PostgresContainer("postgres:16-alpine")
_PG.start()

os.environ.setdefault("DISABLE_WARMUP", "1")     
os.environ.setdefault("SKIP_RANK_MODEL", "1")
os.environ.setdefault("API_STATIC_KEY", "coall")
os.environ.setdefault("JWT_SECRET", "coall")
os.environ.setdefault("REDIS_URL", "memory://")  

os.environ["POSTGRES_USER"] = _PG.username
os.environ["POSTGRES_PASSWORD"] = _PG.password
os.environ["POSTGRES_DB"] = _PG.dbname
os.environ["POSTGRES_HOST"] = _PG.get_container_host_ip()
os.environ["POSTGRES_PORT"] = _PG.get_exposed_port(5432)

engine_url = _engine_url_from_pg(_PG)
os.environ["DATABASE_URL"] = engine_url
os.environ["SQLALCHEMY_DATABASE_URL"] = engine_url

database = importlib.import_module("API.database")
with contextlib.suppress(Exception):
    database.engine.dispose()

database.engine = create_engine(engine_url, future=True)
database.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=database.engine)

models = importlib.import_module("API.models")
models.ensure_ml_schema(database.engine)
models.Base.metadata.create_all(database.engine)

main = importlib.import_module("API.main")
_APP = main.app
_ensure_minimal_app_state_for_api(_APP)


def pytest_sessionfinish(session, exitstatus):
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


@pytest.fixture(autouse=True)
def _db_reset():
    models.ensure_ml_schema(database.engine)
    models.Base.metadata.drop_all(database.engine)
    models.Base.metadata.create_all(database.engine)

    yield


@pytest.fixture
def db_session():
    """Session courte si certains tests veulent manipuler la DB directement."""
    s = database.SessionLocal()
    try:
        yield s
    finally:
        with contextlib.suppress(Exception):
            s.rollback()
        s.close()
