
# AutoML Platform

Taking Pandas Profiling a step further with heavy inspiration from Autosklearn this is a platform that streamlines ypur ml workflow: from data upload to model deployment. 

## Features

### Data Upload & Exploration
- Drag and drop or browse to upload your datasets
- Get instant insights into your data structure
- Identify and handle missing data automatically
- Smart detection of numerical, categorical, and datetime features
- Visualize your data with various plots

### Training and prediction
- Trains multiple ML models
  - Linear Regression
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
  - Support Vector Machines
  - Neural Networks
- Automated hyperparameter optimization
- Robust model evaluation with k-fold CV
- Side-by-side performance metrics

## Architecture

```
┌───────────────────────────────────────────┐
│       Frontend (streamlit ui)             │
└───────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────┐
│              AutoML engine               │
│  performs: EDA                           │
│            Feature Detection             │
│            Preprocessing builder         │
│            Model Selection               │
│            Train + CV                    │
│            SHAP (explainability)         │
└──────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────┐
│              Artifacts (model.pkl)       │
└──────────────────────────────────────────┘
                       ↓
┌───────────────────────────────────────────┐
│   fastAPI Backend (prediction endpoint)   │
└───────────────────────────────────────────┘
```


## Installation

### Clone the Repository

```bash
git clone https://github.com/sonuj04/Auto_ml_platform.git
cd Auto_ml_platform
```

### Create Virtual Environment

```bash
python -m venv venv
#on Linux:
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

##  Usage

### Running the Streamlit App

Run both in separate terminals.

```bash
python -m streamlit run frontend/streamlit_app.py
```

Open your browser to `http://localhost:8501`


Start the API server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`
