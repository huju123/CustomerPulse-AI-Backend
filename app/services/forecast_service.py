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
    Currently: revenue forecast is real (Prophet model output).
    Customer growth, active customers, and churn trend are NOT yet available
    (pending additional models/data from the ML team) — returned as empty lists.
    """
    revenue_points = _load_revenue_forecast()

    return ForecastResponse(
        revenue_forecast=revenue_points,
        customer_growth_forecast=[],       # TODO: pending ML team — no model/data yet
        active_customers_forecast=[],      # TODO: pending ML team — no model/data yet
        churn_trend_forecast=[],           # TODO: pending ML team — no model/data yet
        forecast_generated_on=date.today(),
        forecast_horizon_days=len(revenue_points) * 30,  # approx, since data is monthly
    )