import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import json

# Paths
TRAIN_DATA_PATH = "data/features/features.csv"
NEW_DATA_PATH = "data/new_data.csv"
DRIFT_LOG_PATH = "reports/drift_detection.json"

# Load datasets
print("Loading datasets for drift detection...")
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_new = pd.read_csv(NEW_DATA_PATH)

# Selecting numeric columns only
num_cols = df_train.select_dtypes(include=np.number).columns.tolist()

drift_results = {}

# Perform KS test for each numerical feature
for col in num_cols:
    stat, p_value = ks_2samp(df_train[col].dropna(), df_new[col].dropna())
    drift_results[col] = {
        "ks_statistic": stat,
        "p_value": p_value,
        "drift_detected": p_value < 0.05
    }

# Save results
with open(DRIFT_LOG_PATH, "w") as f:
    json.dump(drift_results, f, indent=4)

print("Drift detection complete! Results saved in reports/drift_detection.json.")