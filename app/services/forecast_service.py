import pandas as pd
from pathlib import Path
from datetime import date
from app.models.schemas import ForecastResponse, ForecastPoint

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REVENUE_FORECAST_CSV = BASE_DIR / "data" / "revenue_forecast.csv"


def _load_revenue_forecast() -> list[ForecastPoint]:
    """
    Load precomputed Prophet revenue forecast.
    Negative values are clipped to 0 — revenue cannot be negative in reality;
    this is a known Prophet artifact on wide confidence intervals, not a backend bug.
    """
    df = pd.read_csv(REVENUE_FORECAST_CSV)

    points = []
    for _, row in df.iterrows():
        forecast_date = pd.to_datetime(row["ds"]).date()
        predicted_value = max(float(row["yhat"]), 0.0)  # clip negative revenue to 0

        points.append(
            ForecastPoint(
                date=forecast_date,
                predicted_value=round(predicted_value, 2),
                lower_bound=None,
                upper_bound=None,
            )
        )
    return points


def get_forecast() -> ForecastResponse:
    """
    Returns business forecasts.
    Revenue forecast: real Prophet model output.
    Customer growth, active customers, churn trend: synthetic data pending ML model development.
    """
    revenue_points = _load_revenue_forecast()
    num_points = len(revenue_points)
    
    # Generate synthetic customer growth forecast (trending up with noise)
    customer_growth = [
        ForecastPoint(
            date=revenue_points[i].date,
            predicted_value=round(50 + (i * 8) + (i % 3) * 12, 2),
            lower_bound=None,
            upper_bound=None,
        )
        for i in range(num_points)
    ]
    
    # Generate synthetic active customers forecast (steady growth)
    active_customers = [
        ForecastPoint(
            date=revenue_points[i].date,
            predicted_value=round(200 + (i * 15) + (i % 4) * 20, 2),
            lower_bound=None,
            upper_bound=None,
        )
        for i in range(num_points)
    ]
    
    # Generate synthetic churn trend forecast (declining churn rate)
    churn_trend = [
        ForecastPoint(
            date=revenue_points[i].date,
            predicted_value=round(max(8.5 - (i * 0.4), 1.0), 2),
            lower_bound=None,
            upper_bound=None,
        )
        for i in range(num_points)
    ]
    
    return ForecastResponse(
        revenue_forecast=revenue_points,
        customer_growth_forecast=customer_growth,
        active_customers_forecast=active_customers,
        churn_trend_forecast=churn_trend,
        forecast_generated_on=date.today(),
        forecast_horizon_days=len(revenue_points) * 30,
    )