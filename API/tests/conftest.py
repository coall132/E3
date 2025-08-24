import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from fastapi.testclient import TestClient
from API import models               
from API import database
from API.main import app 
from sqlalchemy import Table, Column, Integer, Text

def table_etab():
    Table("etab",
        models.Base.metadata,
        Column("id_etab", Integer, primary_key=True),
        Column("nom", Text),    
        extend_existing=True,
    )
    

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
    models.ensure_ml_schema(eng)
    table_etab()
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

@pytest.fixture()
def client(db_session, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
    monkeypatch.setenv("API_STATIC_KEY", "coall")

    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[database.get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()