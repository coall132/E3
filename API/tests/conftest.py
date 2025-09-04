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
def _db_clean():
    with database.engine.begin() as conn:
        tbls = [t for t in models.Base.metadata.sorted_tables]
        if tbls:
            names = ", ".join(f'"{t.name}"' for t in tbls)
            conn.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
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
