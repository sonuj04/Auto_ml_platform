import pandas as pd

def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    unique_vals = df[target].nunique()

    if df[target].dtype == "object" or unique_vals <= 20:
        return "classification"
    return "regression"


def detect_feature_types(df: pd.DataFrame, target: str):
    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if target in numeric_features:
        numeric_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)

    return numeric_features, categorical_features
