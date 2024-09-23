from fastapi import FastAPI  # Importation de FastAPI pour créer une API web
from pydantic import BaseModel  # BaseModel est utilisé pour définir des schémas d'entrée et de validation de données
from tensorflow.keras.models import load_model  # Importation pour charger un modèle Keras
import uvicorn  # Serveur ASGI pour exécuter l'API
import mlflow.keras  # Permet de charger des modèles Keras enregistrés via MLflow
import pandas as pd  # Bibliothèque utilisée pour la manipulation des données
import os  # Module pour interagir avec le système d'exploitation (utile pour la gestion de fichiers)
from pathlib import Path  # Gestion des chemins de fichiers

# Initialisation de l'application FastAPI
app = FastAPI()

# Point de terminaison GET simple pour tester si l'API fonctionne
@app.get("/")
def greet():
    return {"message": "bonjour"}

# Définition de la classe qui modélise l'entrée de l'utilisateur, ici un texte simple
class InputText(BaseModel):
    text: str

# Fonction pour charger un modèle localement (modèle Keras sauvegardé en local)
def load():
    model_path = "model.keras"  # Chemin vers le fichier modèle
    print("\n", model_path)  # Impression du chemin pour vérification
    model = load_model(model_path)  # Chargement du modèle Keras
    return model

# Chargement du modèle au démarrage de l'application
model = load()

# Code alternatif commenté pour charger un modèle à partir de MLflow, en fonction du système de suivi d'expériences.
# os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'  # Définit l'URI du serveur MLflow
# logged_model = f'runs:/c1181b67241b4d92a633b0a67458fbb4/model'  # Spécifie le chemin du modèle stocké dans un run MLflow
# model = mlflow.keras.load_model(logged_model)  # Charge le modèle depuis MLflow
# model.save('test.keras')  # Sauvegarde le modèle chargé localement

# Prétraitement basique : mise en minuscule du texte d'entrée
def preprocess(tweet):
    tweet = [t.lower() for t in tweet]  # Convertit le texte en minuscules
    return tweet

# Point de terminaison POST pour prédire la polarité du texte
@app.post("/predict")
def predict_label(input_text: InputText):
    text = input_text.text  # Récupère le texte fourni par l'utilisateur
    clean_tweet = preprocess([text])  # Applique le prétraitement
    prediction = model.predict(pd.Series(clean_tweet))  # Fait la prédiction avec le modèle chargé

    # Classement du sentiment selon la prédiction
    if prediction[0] > 0.5 :
        sentiment = 'POSITIF'
    if prediction[0] <= 0.5 :
        sentiment = 'NEGATIF'
    
    # Retourne la prédiction et le sentiment associé
    return {"prediction":  ' '.join([str(prediction[0][0]), sentiment])}

# exécuter l'application via uvicorn en mode local
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# Note pour exécuter : uvicorn main:app --reload
