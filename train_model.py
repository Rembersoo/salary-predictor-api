# train_model.py  — using real Kaggle data
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load the real dataset ──────────────────────────────────────
df = pd.read_csv('DataScience_salaries_2025.csv')
print(f"Loaded {len(df)} rows")

# ── Use only the useful columns ────────────────────────────────
df = df[['job_title', 'experience_level', 'employment_type',
         'company_size', 'remote_ratio', 'salary_in_usd']].dropna()

print(f"After cleaning: {len(df)} rows")
print(f"Salary range: ${df['salary_in_usd'].min():,.0f} — ${df['salary_in_usd'].max():,.0f}")

# ── Encode text columns into numbers ──────────────────────────
encoders = {}
for col in ['job_title', 'experience_level', 'employment_type', 'company_size']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"{col} values: {list(le.classes_)}")

# ── Features and target ────────────────────────────────────────
feature_cols = ['job_title_enc', 'experience_level_enc',
                'employment_type_enc', 'company_size_enc', 'remote_ratio']
X = df[feature_cols]
y = df['salary_in_usd']

# ── Split: 80% train, 20% test ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train the model ────────────────────────────────────────────
print("\nTraining model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2  = r2_score(y_test, preds)
print(f"\nModel performance on test set:")
print(f"  MAE  : ${mae:,.0f}  (average prediction error)")
print(f"  R²   : {r2:.3f}    (1.0 = perfect, 0 = random)")

# ── Save model + encoders ──────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoders': encoders}, f)

print("\nSaved to model.pkl — ready to serve!")