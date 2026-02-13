import pandas as pd

def run_eda(df: pd.DataFrame, target: str) -> dict:
    report = {}

    report["shape"] = df.shape
    report["columns"] = df.columns.tolist()
    report["dtypes"] = df.dtypes.astype(str).to_dict()
    report["missing_percentage"] = (
        df.isnull().mean() * 100
    ).round(2).to_dict()

    if target in df.columns:
        report["target_unique_values"] = df[target].nunique()

    return report

