from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import churn, forecast, dashboard, feature_importance

app = FastAPI(
    title="CustomerPulse AI - Backend",
    description="Predictive Customer Churn & Business Forecasting API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(churn.router)
app.include_router(forecast.router)
app.include_router(dashboard.router)
app.include_router(feature_importance.router)


@app.get("/", tags=["Root"])
def root():
    return {"message": "CustomerPulse AI Backend is running 🚀"}