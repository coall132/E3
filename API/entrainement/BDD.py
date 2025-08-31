import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from typing import Dict, Optional
from dotenv import load_dotenv

# Charger les variables d'environnement une seule fois au démarrage.
load_dotenv()

def get_db_engine() -> Optional[Engine]:
    """
    Crée un moteur de connexion SQLAlchemy à partir des variables d'environnement.

    La connexion est "lazy", elle ne sera réellement testée que lors de la
    première requête.

    Returns:
        Un objet Engine de SQLAlchemy si la configuration est valide, sinon None.
    
    Raises:
        ValueError: Si une des variables d'environnement requises est manquante.
    """
    db_user = os.getenv('POSTGRES_USER')
    db_password = os.getenv('POSTGRES_PASSWORD')
    db_name = os.getenv('POSTGRES_DB')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5433')      

    if not all([db_user, db_password, db_name]):
        raise ValueError("Les variables d'environnement POSTGRES_USER, POSTGRES_PASSWORD, et POSTGRES_DB doivent être définies.")

    try:
        db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        print(f"Erreur lors de la configuration du moteur de BDD : {e}")
        return None

def extract_tables_to_dfs(engine: Engine) -> Dict[str, pd.DataFrame]:
    """
    Extrait les tables du schéma public dans un dictionnaire de DataFrames pandas.

    Args:
        engine: L'objet Engine de SQLAlchemy à utiliser pour la connexion.

    Returns:
        Un dictionnaire où les clés sont les noms des tables et les valeurs
        sont les DataFrames correspondants.
    """
    if not engine:
        return {}
        
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names(schema='public')
        
        if not table_names:
            print("⚠️ Aucune table trouvée dans le schéma 'public'.")
            return {}
            
        dfs = {name: pd.read_sql_table(name, engine) for name in table_names}
        print(f"✅ {len(dfs)} tables extraites : {', '.join(dfs.keys())}")
        return dfs
    except Exception as e:
        print(f"Erreur lors de l'extraction des tables : {e}")
        return {}