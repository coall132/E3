import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mydb")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost") 
POSTGRES_PORT = os.getenv("POSTGRES_PORT","5432")

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

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
    