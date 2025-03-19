import pandas as pd


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

df.to_csv("data/features.csv", index=False)