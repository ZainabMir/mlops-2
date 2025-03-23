import joblib
import optuna
import numpy as np
import pandas as pd
import json
import os
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
FEATURES_PATH = "data/features/features.csv"
MODEL_LOAD_PATH = "models/best_model.pkl"
TUNING_LOG_PATH = "reports/hyperparam_tuning.log"
BEST_PARAMS_PATH = "reports/best_hyperparams.json"

# Load Dataset
print("Loading dataset...")
df = pd.read_csv(FEATURES_PATH)

# Check for missing values
print("Checking for missing/infinite values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Define feature columns (excluding the target variable 'income')
feature_cols = [col for col in df.columns if col != "income"]
X = df[feature_cols]
y = df["income"]

# Encode target variable
print("Encoding target variable...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert `<=50K` and `>50K` to 0 and 1

# Encode categorical features using Label Encoding
print("Encoding categorical features...")
categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Handle missing values
print("Handling missing values...")
X.fillna(X.median(numeric_only=True), inplace=True)

# Split dataset
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load best AutoML model
print("Loading the best AutoML model...")
best_model = joblib.load(MODEL_LOAD_PATH)

# Extract best model type
if not isinstance(best_model, HistGradientBoostingClassifier):
    raise ValueError("Loaded model is not HistGradientBoostingClassifier. Ensure AutoML selected the correct model.")

# Define Optuna objective function
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 1000, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 100),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-10, 1e-1, log=True),
    }
    tuned_model = HistGradientBoostingClassifier(**params, random_state=42)
    tuned_model.fit(X_train, y_train)
    preds = tuned_model.predict(X_test)
    return accuracy_score(y_test, preds)

# Run Optuna tuning
print("Starting hyperparameter tuning with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Save best parameters
best_params = study.best_params
with open(BEST_PARAMS_PATH, "w") as f:
    json.dump(best_params, f, indent=4)

# Log tuning results
print("Saving hyperparameter tuning logs...")
with open(TUNING_LOG_PATH, "w") as f:
    for trial in study.trials:
        f.write(f"Trial {trial.number}: {trial.params} -> Accuracy: {trial.value}\n")

print(f"Hyperparameter tuning logs saved at {TUNING_LOG_PATH}")
print(f"Best parameters saved at {BEST_PARAMS_PATH}")
