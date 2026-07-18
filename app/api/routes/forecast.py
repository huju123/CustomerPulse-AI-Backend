from fastapi import APIRouter
from datetime import date, timedelta
from app.models.schemas import ForecastResponse, ForecastPoint

router = APIRouter(
    prefix="/forecast",
    tags=["Forecasting"]
)


@router.get("", response_model=ForecastResponse)
def get_forecast():
    """
    Returns forecasted business metrics (revenue, customer growth, active customers, churn trend).
    Currently returns dummy data — will be wired to Prophet forecasting service.
    """
    # TODO: Replace with real Prophet model output via forecast_service.py
    today = date.today()
    dummy_points = [
        ForecastPoint(
            date=today + timedelta(days=i),
            predicted_value=1000 + (i * 25),
            lower_bound=950 + (i * 20),
            upper_bound=1050 + (i * 30)
        )
        for i in range(7)
    ]

    return ForecastResponse(
        revenue_forecast=dummy_points,
        customer_growth_forecast=dummy_points,
        active_customers_forecast=dummy_points,
        churn_trend_forecast=dummy_points,
        forecast_generated_on=today,
        forecast_horizon_days=7
    )