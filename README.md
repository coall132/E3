# E3
Important !!! Pour faire fonctioner ce repo, installer le repo E1_fin et suiver la suite d'action susdite dans le read_me, il faut
              également que le docker du E1_fin tourne pour faire fonctioner ce repo

Pour que tout fonctionne plusieurs étape sont à prévoir :
    1 - Pour commencer créer vous un webhook discord de cette façon :
        A - Créer un serveur discord
        B - Aller dans les paramètre d'un salon de votre serveur
        C - Aller dans intégration puis webhook
        D - Nouveau Webhook
        E - Clicker sur le webhook créer puis clicker sur copier l'url du webhook


    2 - Ensuite cloner le repo github en local ou sur un serveur

    3 - Aller dans le dossier cloner et créer un fichier .env contenant :
        GOOGLE_KEY= (votre clé API)
        AWS_ACCESS_KEY_ID= (votre user minIO)
        AWS_SECRET_ACCESS_KEY= (votre mdp MinIO)
        POSTGRES_USER=(votre user postgres)
        POSTGRES_PASSWORD=(votre mdp postgres)
        POSTGRES_DB= mydb
        DATABASE_URL='postgresql://(votre user postgres):(votre mdp postgres)@localhost:5432/mydb'
        POSTGRES_HOST='postgres'
        POSTGRES_PORT='5432'
        SECRET=(clé pour le hashage)
        DISCORD_WEBHOOK_URL=(votre webhook discord)
        API_STATIC_KEY=(votre clé statique)
        JWT_SECRET=(votre clé de hashage de token JWT)

        ARTIFACTS_DIR="artifacts"
        PREPROC_PATH="artifacts/preproc_items.joblib"
        RANK_MODEL_PATH="artifacts/rank_model.joblib"

        MLFLOW_BACKEND_STORE_URI= "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
        MLFLOW_ARTIFACTS_DESTINATION= "file:///mlruns2"
        E2E="0"
    
    4 - Pour le CI/CD :
        A - Créer un compte Docker Hub
            1 - Aller dans setting du compte
            2 - Puis personal access token
            3 - Generate a new token avec Read,Write,Delete comme permission
            4 - Copier le token
        B - Cloner l'entiereté vers un autre repo github
        C - Acceder au parametre puis à secret et variable puis action
        D - Creer ces secret :
            API_STATIC_KEY avec (la clé que vous avait choisi plus haut)
            DISCORD_WEBHOOK_URL avec (votre webhook discord)
            DOCKERHUB_TOKEN avec (votre token docker hub)
            DOCKERHUB_USERNAME avec (votre username DockerHub)
            JWT_SECRET avec (votre clé de hashage de token JWT)
        E - Acceder à action puis runner (toujours dans les setting de votre repo)
        F - Suiver le tuto github pour créer un runner sur votre machine
        G - Créer sur votre PC un dossier %USERPROFILE%\app
        H - aller dans le dossier clonner (celui du début) et lancer la commande :
            Copy-Item -Force .\docker-compose.yml "$dst\docker-compose.yml"
            Copy-Item -Force .\.env              "$dst\.env"
            New-Item -ItemType Directory -Force -Path $dst | Out-Null
            $dst = "$env:USERPROFILE\app"
    
    5 - Pour simplement lancer le projet sur votre machine :
        A - Aller dans votre dossier clonner (a la racine) et lancer docker compose up --build -d

L'API est accessible au port 8001
Le client au port 8501
Mlflow au port 5001
    