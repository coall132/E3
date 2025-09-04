# conftest.py
"""
Postgres éphémère via testcontainers + override propre de l'engine SQLAlchemy.
Les tests existants (client_realdb) restent inchangés.
"""
import os
import contextlib
import importlib
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from fastapi.testclient import TestClient


def _engine_url_from_pg(pg: PostgresContainer) -> str:
    """Force le driver psycopg2 pour SQLAlchemy."""
    url = pg.get_connection_url()  # ex: postgresql://test:test@127.0.0.1:xxxxx/test
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


@pytest.fixture(scope="session")
def pg():
    """Démarre un Postgres jetable pour toute la session de tests."""
    container = PostgresContainer("postgres:16-alpine")
    container.start()
    try:
        yield container
    finally:
        with contextlib.suppress(Exception):
            container.stop()


@pytest.fixture(scope="session")
def app(pg):
    """
    Configure l'environnement + l'engine AVANT d'importer l'app FastAPI.
    """
    # Env pour l'app
    os.environ.setdefault("DISABLE_WARMUP", "1")     # pas de charge ML pendant les tests
    os.environ.setdefault("SKIP_RANK_MODEL", "1")
    os.environ.setdefault("API_STATIC_KEY", "coall")
    os.environ.setdefault("JWT_SECRET", "coall")
    os.environ.setdefault("REDIS_URL", "memory://")  # rate limiting en mémoire (pas de Redis requis)

    # Expose aussi les variables PG (si ton code en dépend)
    os.environ["POSTGRES_USER"] = pg.username
    os.environ["POSTGRES_PASSWORD"] = pg.password
    os.environ["POSTGRES_DB"] = pg.dbname
    os.environ["POSTGRES_HOST"] = pg.get_container_host_ip()
    os.environ["POSTGRES_PORT"] = pg.get_exposed_port(5432)

    # URL SQLAlchemy
    engine_url = _engine_url_from_pg(pg)
    os.environ["DATABASE_URL"] = engine_url
    os.environ["SQLALCHEMY_DATABASE_URL"] = engine_url

    # Override explicite de l'engine et SessionLocal
    database = importlib.import_module("API.database")
    with contextlib.suppress(Exception):
        database.engine.dispose()

    database.engine = create_engine(engine_url, future=True)
    database.SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=database.engine
    )

    # Crée schéma/tables AVANT d'importer l'app
    models = importlib.import_module("API.models")
    models.ensure_ml_schema(database.engine)
    models.Base.metadata.create_all(database.engine)

    # Maintenant on peut charger l'app (qui importera l'engine override)
    main = importlib.import_module("API.main")
    return main.app


@pytest.fixture
def client_realdb(app):
    """Client FastAPI branché sur le Postgres éphémère."""
    return TestClient(app)
