import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

from core.train_pipeline import train

st.set_page_config(page_title="AutoML Platform", layout="wide")

st.title("AutoML Platform")

#upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset preview")
    st.dataframe(df.head())



    st.subheader("Dataset summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values",
                int(df.isnull().sum().sum()))

    # select columns
    st.subheader("Select target and features")

    target_column = st.selectbox("Select target column, make sure target column has no missing values", df.columns)

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column]
    )

    selected_df = df[feature_columns + [target_column]]

    # missing value

    st.subheader("Missing Values (%)")

    missing = selected_df.isnull().mean() * 100
    st.bar_chart(missing)

    # distributions
    numeric_cols = selected_df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        st.subheader("Distributions")

        selected_numeric = st.selectbox(
            "Select Numeric Column to Visualize",
            numeric_cols
        )

        fig, ax = plt.subplots()
        sns.histplot(selected_df[selected_numeric], kde=True, ax=ax)
        st.pyplot(fig)

    # heatmap
    if len(numeric_cols) > 1:
        st.subheader("Heatmap")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            selected_df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)

    # train
    if st.button("Run AutoML Training"):

        try:
            with st.spinner("Training in progress..."):
                selected_df.to_csv("temp_data.csv", index=False)
                train("temp_data.csv", target_column)

            st.success("Training Complete!")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")

# predict
st.divider()
st.header("make predictions")

if os.path.exists("artifacts/model.pkl"):

    st.write("Enter feature values:")

    sample_input = {}

    if uploaded_file:

        for col in feature_columns:
            sample_input[col] = st.text_input(f"{col}")

        if st.button("Predict"):

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"data": sample_input}
            )

            if response.status_code == 200:
                result = response.json()

                if "prediction" in result:
                    st.success(f"Prediction: {result['prediction']}")
                else:
                    st.error(result["error"])
            else:
                st.error("Prediction failed.")
