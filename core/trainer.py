from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import numpy as np


def train_and_select(X, y, preprocessor, models, problem_type):

    best_score = -np.inf
    best_pipeline = None

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        if problem_type == "classification":
            scores = cross_val_score(
                pipeline, X, y, cv=5, scoring="accuracy"
            )
        else:
            scores = cross_val_score(
                pipeline, X, y, cv=5, scoring="r2"
            )

        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_pipeline = pipeline

    best_pipeline.fit(X, y)

    return best_pipeline, best_score
