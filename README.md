# Rate Change Prediction Model

This project aims to build a regression model that predicts the **percentage rate of change** between two numeric values — typically representing an "old" and a "new" value — using machine learning techniques.

## Project Overview

Given synthetic data simulating changes between two numerical values (e.g., price changes, currency rates, KPIs), we train a regression model to estimate the percentage change between them. The process includes:

- Synthetic data generation with controlled randomness
- Feature engineering based on domain-relevant transformations (e.g., ratio, residuals)
- Standardization of input features
- Model training with `RandomForestRegressor`
- Hyperparameter optimization using `RandomizedSearchCV`
- Model evaluation using Mean Squared Error (MSE) and cross-validation
- Deployment-ready model and scaler saved with `joblib`

## Technologies & Libraries

- Python 3.x
- NumPy
- Pandas
- Scikit-learn (Random Forest, preprocessing, model selection)
- Joblib (for model serialization)

## Features Used

The model predicts the **percentage rate change** using the following features:
- `old_value`: Original numerical value
- `new_value`: Updated numerical value
- `residual_value`: Difference between new and old value
- `rate`: Ratio of new value to old value

## Model Performance

The model is trained on 8,000 synthetic samples and evaluated using:
- **Test MSE**
- **5-Fold Cross-Validation MSE**

These metrics help ensure the model generalizes well and avoids overfitting.

## How It Works

After training, you can use the `predictValue(old, new)` function to predict the expected percentage change. The model uses the scaled input and trained estimator to return a predicted rate.

## Model Export

Trained components are saved as:
- `bestModel.pkl` – The trained Random Forest model
- `scaler.pkl` – The StandardScaler used for input normalization

These files can later be loaded for predictions in production systems or APIs.

## Use Case Examples

This framework can be adapted for real-world scenarios such as:
- Financial rate prediction
- Inventory price tracking
- Sensor reading comparisons
- KPI evolution analysis
