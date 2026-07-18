from fastapi import FastAPI
from app.api.routes import churn, forecast, dashboard, feature_importance

app = FastAPI(
    title="CustomerPulse AI - Backend",
    description="Predictive Customer Churn & Business Forecasting API",
    version="1.0.0"
)

# Register all route modules
app.include_router(churn.router)
app.include_router(forecast.router)
app.include_router(dashboard.router)
app.include_router(feature_importance.router)


@app.get("/", tags=["Root"])
def root():
    return {"message": "CustomerPulse AI Backend is running 🚀"}