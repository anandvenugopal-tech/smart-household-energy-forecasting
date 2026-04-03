from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
model = joblib.load('models/randomforest_model.pkl')

class EnergyInput(BaseModel):
    
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: float
    Sub_metering_2: float
    Sub_metering_3: float
    Month: int
    Day: int
    Weekday: int
    Hour: int
    lag_1: float
    lag_60: float
    lag_1440: float
    rolling_60: float
    rolling_1440: float

@app.get("/home")
def home():
    return {"message": "Energy Forecast API Running"}

@app.post("/predict")
def predict(data: EnergyInput):
    df = pd.DataFrame([dict(data)])
    
    prediction = model.predict(df)

    return {"Prediction": prediction[0]}

