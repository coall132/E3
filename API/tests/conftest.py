import os
os.environ.setdefault("DISABLE_WARMUP", "1")  # avant tout import de main/app

import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, Text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

# Permet de fonctionner avec un projet packagé "API.*" ou flat.
try:
    from API import models as models
    from API import database as db
    from API.main import app as fastapi_app
    import API.CRUD as CRUD
    from API.benchmark_2_0 import pick_anchors_from_df
except ImportError:
    from API import models
    from API import database as db
    from API.main import app as fastapi_app
    import API.CRUD as CRUD
    from API.benchmark_2_0 import pick_anchors_from_df


class _StubSentModel:
    """Stub d'embedding pour éviter de charger un gros modèle pendant les tests."""
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        return [np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32) for _ in texts]


def _ensure_min_tables():
    # Table minimale si ton metadata ne la déclare pas déjà
    Table(
        "etab",
        models.Base.metadata,
        Column("id_etab", Integer, primary_key=True),
        Column("nom", Text),
        extend_existing=True,
    )


# ---------- DB container (intégration) ----------
@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("postgres:16-alpine") as pg:
        pg.start()
        yield pg


@pytest.fixture(scope="session")
def engine(pg_container):
    url = pg_container.get_connection_url().replace(
        "postgresql://", "postgresql+psycopg2://"
    )
    eng = create_engine(url, pool_pre_ping=True)

    # si tu as une fonction utilitaire pour créer un schéma ML
    if hasattr(models, "ensure_ml_schema"):
        models.ensure_ml_schema(eng)

    _ensure_min_tables()
    models.Base.metadata.create_all(bind=eng)
    yield eng


@pytest.fixture()
def db_session(engine):
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ---------- état applicatif + client ----------
@pytest.fixture(autouse=True)
def _setup_app_state():
    n = 6
    df = pd.DataFrame({
        "id_etab": range(1, n + 1),
        "rating": [4.2, 3.8, 4.6, 4.9, 3.1, 4.0],
        "priceLevel": [1, 2, 3, 2, 1, 4],
        "latitude": [47.39] * n,
        "longitude": [0.68] * n,
        "editorialSummary_text": [
            "bistronomie locale", "pizzeria", "cuisine créative", "gastro", "burger", "sushis"
        ],
        "start_price": [10, 15, 25, 30, 9, 20],
        "code_postal": ["37000", "37000", "37100", "37200", "37000", "37100"],
        "delivery": [True, False, True, False, True, False],
        "servesVegetarianFood": [True, True, False, True, False, False],
        "desc_embed": [np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32)] * n,
        "rev_embeds": [None] * n,
        "ouvert_lundi_midi": [1, 1, 0, 1, 0, 1],
    })

    fastapi_app.state.DF_CATALOG = df
    fastapi_app.state.SENT_MODEL = _StubSentModel()
    fastapi_app.state.PREPROC = None
    fastapi_app.state.ML_MODEL = None
    fastapi_app.state.ANCHORS = pick_anchors_from_df(df, n=4)
    fastapi_app.state.FEATURE_COLS = []
    yield


@pytest.fixture
def client(db_session, monkeypatch):
    # Config tests
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
    monkeypatch.setenv("API_STATIC_KEY", "coall")

    # override DB dependency pour utiliser la session test
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    fastapi_app.dependency_overrides[db.get_db] = override_get_db

    # bypass auth pour tests unitaires (si tu veux tester l’auth, fais-le dans un autre test sans cet override)
    fastapi_app.dependency_overrides[CRUD.get_current_subject] = lambda: "user:1"

    with TestClient(fastapi_app) as c:
        yield c

    fastapi_app.dependency_overrides.clear()