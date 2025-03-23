import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
FEATURES_PATH = "data/features/features.csv"
df = pd.read_csv(FEATURES_PATH)

# taking a sample 
df = df.sample(1000)

# Identify categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Apply Label Encoding for categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use if needed

# Extract features (drop target 'income')
X = df.drop(columns=["income"]).astype(np.float32)

# Load trained model
MODEL_LOAD_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_LOAD_PATH)

# Create SHAP Explainer (Permutation Explainer for non-supported models)
explainer = shap.Explainer(model.predict, X, algorithm="permutation")
shap_values = explainer(X)

# Feature Importance Plot
shap.summary_plot(shap_values, X)

# Save the plot
plt.savefig("reports/shap_feature_importance.png")
plt.close()

print("SHAP feature importance plot saved at reports/shap_feature_importance.png")
