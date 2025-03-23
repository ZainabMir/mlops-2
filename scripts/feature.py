import pandas as pd
import os 
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                "hours-per-week", "native-country", "income"]

df = pd.read_csv("data/adult.csv", names=column_names, na_values=" ?", skipinitialspace=True)

df.info()

# Feature 1: Income per hour
df["income_per_hour"] = df["capital-gain"] / (df["hours-per-week"] + 1)

# Feature 2: Total capital wealth
df["total_capital"] = df["capital-gain"] - df["capital-loss"]

# Feature 3: Age Binning (Young, Middle-Aged, Senior)
df["age_group"] = pd.cut(df["age"], bins=[0, 25, 45, 65, 100], labels=["Young", "Middle-Aged", "Senior", "Elder"])

# Feature 4: Workclass grouping (Simplify work categories)
df["workclass_grouped"] = df["workclass"].replace(
    {"Self-emp-not-inc": "Self-Employed", "Self-emp-inc": "Self-Employed",
     "Local-gov": "Government", "State-gov": "Government", "Federal-gov": "Government",
     "Without-pay": "Other", "Never-worked": "Other"}
)

# Feature 5: Is Government Employee?
df["is_gov_employee"] = df["workclass"].isin(["Local-gov", "State-gov", "Federal-gov"]).astype(int)

# Feature 6: Family size (based on marital status)
df["is_married"] = df["marital-status"].apply(lambda x: 1 if "Married" in x else 0)

print(df.head())

# Drop 'fnlwgt' (not useful for prediction)
df.drop(columns=["fnlwgt"], inplace=True)

# ===== Data Preprocessing =====
# Convert categorical variables using Label Encoding
categorical_cols = ["workclass_grouped", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country", "age_group"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Convert target variable (income: <=50K -> 0, >50K -> 1)
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "income_per_hour", "total_capital"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

 
os.makedirs('data/features', exist_ok=True)  
df.to_csv("data/features/features.csv")

print("Feature Engineering & Preprocessing completed. Preprocessed data saved at data/features/features_preprocessed.csv")
