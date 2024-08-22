import pytest
from fastapi.testclient import TestClient
from main import app, preprocess, InputText  # import your app and necessary components
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization  # Import the TextVectorization layer
import uvicorn
import mlflow.keras
import pandas as pd
import os

import  main


# Create a TestClient using the FastAPI app
client = TestClient(app)

@pytest.fixture(scope="module")
def setup_model():
    model = main.load()
    # Load the model or any required setup
    yield model


def test_second_layer_is_textvectorization(setup_model):
    print("\n\n *****************************************") 
    print("TEST 001 : test_second_layer_is_textvectorization() ") 
    tested_model = setup_model  # Get the model from the fixture
    tested_model.summary()
    assert isinstance(tested_model.layers[0], TextVectorization), "There is no TextVectorization layer"



def test_predict_negative(setup_model):
    print("\n\n *****************************************") 
    print("TEST 002 : test_predict_negative() ") 
    # Define a positive input
    input_text = InputText(text="terrible")

    # Send POST request to /predict
    response = client.post("/predict", json=input_text.dict())
    result = response.json()
    print(result["prediction"])    
    # Test the prediction output
    assert float(result["prediction"].split()[0]) < 0.1 , "The score is too high for negative validation" 


def test_predict_positive(setup_model):
    print("\n\n *****************************************") 
    print("TEST 003 : test_predict_positive() ") 
    # Define a negative input
    input_text = InputText(text="awesome")

    # Send POST request to /predict
    response = client.post("/predict", json=input_text.dict())
    result = response.json()
    print('Positive testing : ')
    print(result["prediction"])
    # Test the prediction output
    assert float(result["prediction"].split()[0]) > 0.9 , "The score is too low for negative validation" 
