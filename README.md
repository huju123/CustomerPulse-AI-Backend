# CustomerPulse AI — Backend

Backend service for **CustomerPulse AI**, a predictive customer churn and business forecasting platform.

## Tech Stack
- FastAPI
- Python
- Uvicorn
- Pydantic
- Scikit-learn / XGBoost (Churn Prediction)
- Prophet (Forecasting)
- SHAP (Explainable AI)

## Setup

1. Create virtual environment:
   \`\`\`bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Run the server:
   \`\`\`bash
   python run.py
   \`\`\`

4. Visit:
   - API root: http://127.0.0.1:8000
   - Swagger docs: http://127.0.0.1:8000/docs

## API Endpoints (Planned)
- `POST /predict-churn`
- `GET /forecast`
- `GET /dashboard-summary`
- `GET /feature-importance`