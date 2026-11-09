# Student Performance Prediction System

## Problem Statement
This prototype helps identify students who are at risk of failing their final Mathematics grade early in the academic cycle.

## Dataset & Target Definition
- **Dataset:** UCI Machine Learning Repository (ID 320 - Student Performance Dataset).
- **Target Rule:** Binary classification based on G3 (`1 = AT RISK` if G3 < 10, `0 = NOT AT RISK` if G3 >= 10). G3 is excluded from input features to avoid target leakage.

## Installation and Setup Instructions
1. Install required packages: `pip install -r requirements.txt`
2. Preprocess and train model: `python train_model.py`
3. Launch Streamlit interface: `streamlit run app.py`

## Project Limitations
Model performance depends on historical dataset context (Portuguese secondary schools) and may require retraining before applying to external institutions.
