import joblib
import pandas as pd
import os


class Predictor:

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists("artifacts/model.pkl"):
            self.model = joblib.load("artifacts/model.pkl")
        else:
            self.model = None

    def predict(self, data: dict):

        # Reload model in case it was trained after startup
        self.load_model()

        if self.model is None:
            raise Exception("Model not trained yet.")

        df = pd.DataFrame([data])
        prediction = self.model.predict(df)

        return prediction.tolist()
