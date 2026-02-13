from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    KFold
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error
)
import numpy as np
import pandas as pd


def get_param_grids(model_name, problem_type):

    if "random_forest" in model_name:
        return {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20]
        }

    if "gradient_boosting" in model_name:
        return {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1]
        }

    if "svm" in model_name:
        return {
            "model__C": [0.1, 1, 10]
        }

    if "neural_network" in model_name:
        return {
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__alpha": [0.0001, 0.001]
        }

    return {}


def train_and_select(X, y, preprocessor, models, problem_type):

    results = []
    best_score = -np.inf
    best_pipeline = None

    if problem_type == "classification":
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "f1_weighted"
    else:
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "r2"

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        param_grid = get_param_grids(name, problem_type)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_iter=5,
            n_jobs=-1
        )

        search.fit(X, y)

        best_model = search.best_estimator_
        best_cv_score = search.best_score_

        results.append({
            "model": name,
            "cv_score": best_cv_score
        })

        if best_cv_score > best_score:
            best_score = best_cv_score
            best_pipeline = best_model

    results_df = pd.DataFrame(results)

    return best_pipeline, best_score, results_df
