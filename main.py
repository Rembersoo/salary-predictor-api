# main.py — updated for real Kaggle data
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load model at startup
with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)
    model    = saved['model']
    encoders = saved['encoders']

app = FastAPI(title="Data Science Salary Predictor")

class SalaryRequest(BaseModel):
    job_title: str           # e.g. "Data Scientist"
    experience_level: str    # EN | MI | SE | EX
    employment_type: str     # FT | PT | CT | FL
    company_size: str        # S | M | L
    remote_ratio: int        # 0, 50, or 100

@app.get("/")
def root():
    return {"status": "running", "model": "Data Science Salary Predictor"}

@app.get("/options")
def options():
    """Returns all valid input values — useful for your frontend"""
    return {
        col: list(enc.classes_)
        for col, enc in encoders.items()
    }

@app.post("/predict")
def predict(req: SalaryRequest):
    try:
        enc = encoders
        features = np.array([[
            enc['job_title'].transform([req.job_title])[0],
            enc['experience_level'].transform([req.experience_level])[0],
            enc['employment_type'].transform([req.employment_type])[0],
            enc['company_size'].transform([req.company_size])[0],
            req.remote_ratio
        ]])
        salary = model.predict(features)[0]
        return {
            "predicted_salary_usd": round(float(salary), 2),
            "inputs": req.dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input value: {e}")