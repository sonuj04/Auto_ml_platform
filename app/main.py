from fastapi import FastAPI
from app.schemas import PredictionInput
from app.predict import Predictor

app = FastAPI()

predictor = Predictor()


@app.post("/predict")
def predict(input_data: PredictionInput):
    prediction = predictor.predict(input_data.data)
    return {"prediction": prediction}
