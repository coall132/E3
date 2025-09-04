import API.entrainement.BDD

def main():
    """
    Fonction principale pour orchestrer l'ensemble du pipeline.
    """
    print("🚀 Démarrage du pipeline de benchmark IA...")

    # Étape 1: Connexion à la base de données (appel simplifié)
    engine = API.entrainement.BDD.get_db_engine()
    if not engine:
        print("Arrêt du pipeline en raison d'un échec de connexion à la BDD.")
        return    
    # Étape 2: Extraction des tables en DataFrames
    all_dfs = API.entrainement.BDD.extract_tables_to_dfs(engine)
    if not all_dfs:
        print("Arrêt du pipeline car aucune donnée n'a pu être extraite.")
    return all_dfs

if __name__ == "__main__":
    main()