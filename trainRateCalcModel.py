import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 8000
oldValues = np.random.uniform(0.01, 0.9, num_samples)  # original values
newValues = oldValues * (1 + np.random.uniform(-0.3, 0.3, num_samples))  # new values with ±30% variation

# Create DataFrame
df = pd.DataFrame({
    'old_value': oldValues,
    'new_value': newValues
})

# Calculate derived features
df['changeRate'] = ((df['new_value'] - df['old_value']) / df['new_value']) * 100  # percentage change
df['residual_value'] = df['new_value'] - df['old_value']  # absolute difference
df['rate'] = df['new_value'] / df['old_value']  # ratio of new to old

# Filter out extreme outliers
df = df[np.abs(df['changeRate']) < 100]

# Define features (X) and target (y)
X = df[['old_value', 'new_value', 'residual_value', 'rate']]
y = df['changeRate']

# Scale the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.22, random_state=42)

# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize and search best model with cross-validation
rf = RandomForestRegressor(random_state=42)
search = RandomizedSearchCV(rf, param_grid, cv=3, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1)
search.fit(X_train, y_train)

# Get the best model from the search
bestModel = search.best_estimator_

# Evaluate model on test data
y_pred = bestModel.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n Test Data - MSE: {mse: .6f}")

# Cross-validate the best model
cv_scores = cross_val_score(bestModel, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-np.mean(cv_scores): .6f}")

# Define a function to make predictions using old and new values
def predictValue(oldValue, newValue):
    residual_value = newValue - oldValue
    rate = newValue / oldValue
    data = np.array([[oldValue, newValue, residual_value, rate]])
    scaled_input = scaler.transform(data)
    return bestModel.predict(scaled_input)[0]

# Run a sample prediction
sample_prediction = predictValue(0.043543, 0.032432)
print(f"Predicted rate is: {sample_prediction:.6f}%")

# Save model and scaler for later use
joblib.dump(bestModel, 'bestModel.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved")
