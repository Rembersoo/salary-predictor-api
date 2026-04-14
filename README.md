# Data Science Salary Predictor API

A production-ready machine learning API that predicts salaries for Data Science 
roles based on job title, experience level, employment type, company size, 
and remote ratio.

Built with real-world salary data (2020–2025) from Kaggle — 9,000+ records 
across global companies.

## Live API
🔗https://salary-predictor-api-production-7158.up.railway.app

## Tech Stack
- **ML Model** — Random Forest Regressor (scikit-learn), R² score ~0.84
- **API** — FastAPI with auto-generated Swagger UI
- **Data** — Kaggle: Data Science Job Salaries 2020–2025 (9,355 rows)
- **Containerisation** — Docker
- **Deployment** — Railway (auto-deploy on every git push)
- **Language** — Python 3.11

## API Endpoints
| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | /          | Health check                       |
| GET    | /options   | Returns all valid input values     |
| POST   | /predict   | Predicts salary from job details   |

## Example Request
POST /predict
{
  "job_title": "Data Scientist",
  "experience_level": "SE",
  "employment_type": "FT",
  "company_size": "M",
  "remote_ratio": 100
}

## Example Response
{
  "predicted_salary_usd": 128340.50,
  "inputs": { ... }
}

## Run Locally
git clone https://github.com/YOUR_USERNAME/salary-predictor-api
cd salary-predictor-api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload

## Run with Docker
docker build -t salary-predictor .
docker run -p 8000:8000 -e PORT=8000 salary-predictor

## What I Learned
- Training and evaluating a Random Forest regression model on real data
- Building a REST API with FastAPI (Python equivalent of Spring Boot)
- Containerising a Python ML app with Docker
- Deploying a live public API with CI/CD via Railway + GitHub
