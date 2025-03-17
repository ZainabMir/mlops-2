import os
import pandas as pd
import ydata_profiling as pandas_profiling
import sweetviz as sv

# Ensure the reports directory exists
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# Load Dataset
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                "hours-per-week", "native-country", "income"]
df = pd.read_csv("data/adult.csv", names=column_names, na_values=" ?", skipinitialspace=True)

# Generate Pandas Profiling Report
profile = df.profile_report(title="Adult Income Dataset EDA")
profile_path = os.path.join(reports_dir, "pandas_profiling_report.html")
profile.to_file(profile_path)

# Generate Sweetviz Report
report = sv.analyze(df)
sweetviz_path = os.path.join(reports_dir, "sweetviz_report.html")
report.show_html(sweetviz_path)

print(f"EDA reports generated:\n- {profile_path}\n- {sweetviz_path}")
