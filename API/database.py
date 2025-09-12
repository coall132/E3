import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv()

url = os.getenv("DATABASE_URL")
if not url:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db = os.getenv("POSTGRES_DB", "postgres")
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
engine = create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def extract_table(engine: Engine,table_name: str, schema: str = "public"):
    if engine is None:
        raise ValueError("Engine invalide (None).")

    insp = inspect(engine)
    if not insp.has_table(table_name, schema=schema):
        print(f"[bdd.extract] table introuvable: {schema}.{table_name}")
        return pd.DataFrame()

    try:
        df = pd.read_sql_table(table_name, con=engine, schema=schema)
        return df
    except Exception as e:
        print(f"[bdd.extract] Erreur lecture {schema}.{table_name}: {e}")
        return pd.DataFrame()
    