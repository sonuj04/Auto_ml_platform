from fastapi import FastAPI
from app.schemas import PredictionInput
from app.predict import Predictor

app = FastAPI()

predictor = Predictor()


@app.post("/predict")
def predict(input_data: PredictionInput):

    try:
        prediction = predictor.predict(input_data.data)
        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
