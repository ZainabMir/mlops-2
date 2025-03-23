import joblib
import optuna
import numpy as np
import pandas as pd
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
FEATURES_PATH = "data/features/features.csv"
MODEL_LOAD_PATH = "models/best_model.pkl"
TUNING_LOG_PATH = "reports/hyperparam_tuning.log"
BEST_PARAMS_PATH = "reports/best_hyperparams.json"
DRIFT_LOG_PATH = "reports/drift_detection.log"

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

# Ensure the model is of correct type
if not isinstance(best_model, HistGradientBoostingClassifier):
    raise ValueError("Loaded model is not HistGradientBoostingClassifier. Ensure AutoML selected the correct model.")

# Start MLflow Experiment
mlflow.set_experiment("Model Performance Tracking")

with mlflow.start_run():
    print("Tracking initial AutoML model performance with MLflow...")
    
    # Get predictions
    y_pred = best_model.predict(X_test)
    
    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log Model Metrics
    mlflow.log_metric("AutoML Accuracy", accuracy)
    mlflow.log_metric("AutoML Precision", precision)
    mlflow.log_metric("AutoML Recall", recall)
    mlflow.log_metric("AutoML F1-score", f1)
    
    # Save Model
    mlflow.sklearn.log_model(best_model, "AutoML_HistGradientBoosting")

print("Starting hyperparameter tuning with Optuna...")

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
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Save best parameters
best_params = study.best_params
with open(BEST_PARAMS_PATH, "w") as f:
    json.dump(best_params, f, indent=4)

# Train the best tuned model
print("Training the best tuned model...")
tuned_model = HistGradientBoostingClassifier(**best_params, random_state=42)
tuned_model.fit(X_train, y_train)
tuned_preds = tuned_model.predict(X_test)

# Compute Metrics for Tuned Model
tuned_accuracy = accuracy_score(y_test, tuned_preds)
tuned_precision = precision_score(y_test, tuned_preds)
tuned_recall = recall_score(y_test, tuned_preds)
tuned_f1 = f1_score(y_test, tuned_preds)

# Track Performance in MLflow
with mlflow.start_run():
    print("Logging tuned model performance in MLflow...")
    
    mlflow.log_params(best_params)
    mlflow.log_metric("Tuned Accuracy", tuned_accuracy)
    mlflow.log_metric("Tuned Precision", tuned_precision)
    mlflow.log_metric("Tuned Recall", tuned_recall)
    mlflow.log_metric("Tuned F1-score", tuned_f1)

    mlflow.sklearn.log_model(tuned_model, "Tuned_HistGradientBoosting")

# Log Tuning Results
print("Saving hyperparameter tuning logs...")
with open(TUNING_LOG_PATH, "w") as f:
    for trial in study.trials:
        f.write(f"Trial {trial.number}: {trial.params} -> Accuracy: {trial.value}\n")

print(f"Hyperparameter tuning logs saved at {TUNING_LOG_PATH}")
print(f"Best parameters saved at {BEST_PARAMS_PATH}")

# DRIFT DETECTION
print("Performing drift detection...")

def population_stability_index(expected, actual, bins=10):
    expected_hist, _ = np.histogram(expected, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual, bins=bins, density=True)
    
    expected_hist = np.clip(expected_hist, 1e-10, None)
    actual_hist = np.clip(actual_hist, 1e-10, None)
    
    psi_values = (expected_hist - actual_hist) * np.log(expected_hist / actual_hist)
    psi = np.sum(psi_values)
    
    return psi

drift_scores = {}
for col in X.columns:
    psi_value = population_stability_index(X_train[col], X_test[col])
    drift_scores[col] = psi_value

with open(DRIFT_LOG_PATH, "w") as f:
    for feature, score in drift_scores.items():
        f.write(f"{feature}: PSI={score}\n")

print(f"Drift detection logs saved at {DRIFT_LOG_PATH}")