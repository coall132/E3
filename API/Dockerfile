FROM python:3.9-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier le fichier des dépendances et les installer
COPY requirements_api.txt requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

# 4. Copier tout le code de l'API dans le conteneur
COPY . .

# 5. Exposer le port sur lequel l'API va tourner
EXPOSE 8000

# 6. Commande pour lancer l'API avec Uvicorn au démarrage du conteneur
#    --host 0.0.0.0 permet de rendre l'API accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]