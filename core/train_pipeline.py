import pandas as pd
import joblib
import os

from core.eda import run_eda
from core.feature_detection import (
    detect_problem_type,
    detect_feature_types
)
from core.preprocessing import build_preprocessor
from core.model_selector import get_models
from core.trainer import train_and_select


def train(csv_path, target):

    df = pd.read_csv(csv_path)

    # 🚨 Remove rows where target is missing
    df = df.dropna(subset=[target])

    if df.shape[0] == 0:
        raise ValueError("Target column contains only missing values.")

    print("Running EDA...")
    report = run_eda(df, target)
    print(report)

    problem_type = detect_problem_type(df, target)
    numeric, categorical = detect_feature_types(df, target)

    X = df.drop(columns=[target])
    y = df[target]

    # remove nan columns
    X = X.dropna(axis=1, how="all")

    preprocessor = build_preprocessor(numeric, categorical)
    models = get_models(problem_type)

    best_model, score = train_and_select(
        X, y, preprocessor, models, problem_type
    )

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/model.pkl")

    print(f"Training complete. Best Score: {score}")


if __name__ == "__main__":
    train("data.csv", "target_column")
