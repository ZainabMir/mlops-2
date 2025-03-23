import pandas as pd
import numpy as np
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os


# Load preprocessed features dataset
FEATURES_PATH = "data/features/features.csv"
MODEL_SAVE_PATH = "models/best_model.pkl"
RESULTS_REPORT = "reports/automl_results.md"
JUSTIFICATION_REPORT = "reports/model_selection_justification.md"

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

# Convert categorical features to numeric using Label Encoding
print("Encoding categorical features...")
categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Fill missing values
print("Handling missing values...")
X.fillna(X.median(numeric_only=True), inplace=True)

# Split dataset
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AutoML Model Selection
print("Initializing AutoML model...")
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=1800,  # Total search time in seconds
    per_run_time_limit=200,  # Time limit per model
    ensemble_size=1,  # Select only the best model
)

print("AutoML model training started...")
automl.fit(X_train, y_train)

# Get leaderboard results
models_summary = pd.DataFrame(automl.leaderboard())
models_summary.to_csv("reports/automl_leaderboard.csv", index=False)

print(f"Models: {automl.show_models()}")

# Select the best model
# best_model = automl.show_models()[0][1]
best_model_key = list(automl.show_models().keys())[0]  # Get the first key (e.g., 5)
best_model = automl.show_models()[best_model_key]['sklearn_classifier']
joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"Best model saved at {MODEL_SAVE_PATH}")


# Save model comparison table
with open(RESULTS_REPORT, "w") as f:
    f.write("# AutoML Model Comparison\n\n")
    f.write(models_summary.to_markdown(index=False))

print(f"AutoML results saved at {RESULTS_REPORT}")