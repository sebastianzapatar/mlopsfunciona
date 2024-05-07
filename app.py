# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("modeloApi")

# Define a Pydantic model for input data
class ModeloApiInput(BaseModel):
    Carat_Weight: float
    Cut: str
    Color: str
    Clarity: str
    Polish: str
    Symmetry: str
    Report: str

# Define a Pydantic model for output data
class ModeloApiOutput(BaseModel):
    prediction: float

# Define predict function
@app.post("/predict", response_model=ModeloApiOutput)
def predict(data: ModeloApiInput):
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=df)
    # Return prediction in the defined output model format
    return ModeloApiOutput(prediction=predictions["Label"].iloc[0])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
