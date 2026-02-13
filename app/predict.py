import joblib
import pandas as pd


class Predictor:

    def __init__(self):
        self.model = joblib.load("artifacts/model.pkl")

    def predict(self, data: dict):
        df = pd.DataFrame([data])
        prediction = self.model.predict(df)
        return prediction.tolist()
