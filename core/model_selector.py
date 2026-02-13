from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor
)

def get_models(problem_type: str):

    if problem_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=2000),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "svm": SVC(probability=True),
            "neural_network": MLPClassifier(max_iter=500)
        }

    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "svm": SVR(),
        "neural_network": MLPRegressor(max_iter=500)
    }
