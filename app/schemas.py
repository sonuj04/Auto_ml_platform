from pydantic import BaseModel


class PredictionInput(BaseModel):
    data: dict
