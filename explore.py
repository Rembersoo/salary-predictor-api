# explore.py
import pandas as pd

df = pd.read_csv('DataScience_salaries_2025.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nJob titles (sample):")
print(df['job_title'].value_counts().head(10))
print("\nExperience levels:")
print(df['experience_level'].value_counts())