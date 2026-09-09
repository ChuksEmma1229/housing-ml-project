from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")


class House(BaseModel):
    id: float
    date: str
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float | None = None
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: str
    yr_built: float
    yr_renovated: float
    zipcode: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float


@app.post("/predict")
def predict(data: House):

    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)

    return {
        "prediction": float(prediction[0])
    }