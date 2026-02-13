import shap


def generate_shap_values(pipeline, X_sample):

    model = pipeline.named_steps["model"]

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        return shap_values
    except Exception:
        return None
