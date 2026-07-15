from fastapi import FastAPI

app = FastAPI(
    title="CustomerPulse AI - Backend",
    description="Predictive Customer Churn & Business Forecasting API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "CustomerPulse AI Backend is running 🚀"}