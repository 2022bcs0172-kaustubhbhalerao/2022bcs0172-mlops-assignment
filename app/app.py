from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle
import os

app = FastAPI()

NAME = "Kaustubh Bhalerao"
ROLL_NO = "2022bcs0172"

class PredictRequest(BaseModel):
    features: List[float]
    use_all_features: Optional[bool] = True

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "name": NAME,
        "roll_no": ROLL_NO
    }

@app.post("/predict")
def predict(request: PredictRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = float(sum(request.features) * 1000)
    return {
        "prediction": round(prediction, 2),
        "name": NAME,
        "roll_no": ROLL_NO
    }
