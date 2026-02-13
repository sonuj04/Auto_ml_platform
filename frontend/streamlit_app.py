import streamlit as st
import pandas as pd
import requests
import os

from core.train_pipeline import train


st.set_page_config(page_title="AutoML Platform", layout="wide")

st.title("End-to-End AutoML Platform")

#upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Select Target Column")
    target_column = st.selectbox("Target", df.columns)

    if st.button("Run AutoML Training"):

        with st.spinner("Training in progress..."):

            df.to_csv("temp_data.csv", index=False)
            train("temp_data.csv", target_column)

        st.success("Training Complete!")

# prediction
st.divider()
st.header("Make Predictions")

if os.path.exists("artifacts/model.pkl"):

    st.write("Enter feature values:")

    sample_input = {}

    if uploaded_file:
        for col in df.columns:
            if col != target_column:
                sample_input[col] = st.text_input(f"{col}")

        if st.button("Predict"):

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"data": sample_input}
            )

            if response.status_code == 200:
                st.success(f"Prediction: {response.json()['prediction']}")
            else:
                st.error("Prediction failed.")
