from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


def get_models(problem_type: str):

    if problem_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier()
        }

    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor()
    }
