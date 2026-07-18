from fastapi import APIRouter
from app.models.schemas import ForecastResponse
from app.services import forecast_service

router = APIRouter(
    prefix="/forecast",
    tags=["Forecasting"]
)


@router.get("", response_model=ForecastResponse)
def get_forecast():
    """
    Returns business forecasts. Revenue forecast is powered by a trained Prophet model.
    Other metrics (customer growth, active customers, churn trend) are pending
    additional ML models and will be added once available.
    """
    return forecast_service.get_forecast()