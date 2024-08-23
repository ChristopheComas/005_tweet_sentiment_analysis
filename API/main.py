from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model # type: ignore
import uvicorn
import mlflow.keras
import pandas as pd
import os
from pathlib import Path


os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'


app = FastAPI()


@app.get("/")
def greet():
    return {"message": "bonjour"}

class InputText(BaseModel):
    text: str

# via local
def load():
#    model_path = "API/LSTM.keras"
    model_path = Path("API") / "LSTM.keras"
    print("\n", model_path)
    model = load_model(model_path)
    return model
model = load()

# via mlflow
# logged_model = f'runs:/c1181b67241b4d92a633b0a67458fbb4/model'
# model = mlflow.keras.load_model(logged_model)
# model.save('test.keras')

def preprocess(tweet):
    tweet = [t.lower() for t in tweet]
    return tweet

@app.post("/predict")
def predict_label(input_text: InputText):
    text = input_text.text
    clean_tweet = preprocess([text])
    prediction = model.predict(pd.Series(clean_tweet))

    if prediction[0] > 0.5 :
        sentiment = 'POSITIF'
    if prediction[0] <= 0.5 :
        sentiment = 'NEGATIF'        
    return {"prediction":  ' '.join([str(prediction[0][0]),sentiment])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

    #   to launch uvicorn main:app --reload 