# conftest.py
import os, pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from fastapi.testclient import TestClient

from API.main import app as fastapi_app
from API import database as db



@pytest.fixture
def client_realdb(monkeypatch):
    dsn = os.getenv("DATABASE_URL")
    API_STATIC_KEY = os.getenv("API_STATIC_KEY")
    if not dsn:
        pytest.skip(f"database non défini")
    if "prod" in dsn.lower():
        pytest.skip("Refus de tester sur une base 'prod'")

    monkeypatch.setenv("DISABLE_WARMUP", "0")       
    monkeypatch.setenv("SKIP_RANK_MODEL", "1")      
    engine = create_engine(dsn, pool_pre_ping=True)
    connection = engine.connect()
    trans = connection.begin()

    TestingSession = sessionmaker(bind=connection, autocommit=False, autoflush=False)
    session = scoped_session(TestingSession)

    nested = session.begin_nested()
    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess, trans_):
        if trans_.nested and not trans_._parent.nested:
            sess.begin_nested()

    db.engine = engine
    db.SessionLocal = TestingSession

    def override_get_db():
        try:
            yield session
        finally:
            pass
    fastapi_app.dependency_overrides[db.get_db] = override_get_db

    try:
        with TestClient(fastapi_app) as c:
            yield c
    finally:
        session.remove()
        trans.rollback()   
        connection.close()
        fastapi_app.dependency_overrides.clear()
