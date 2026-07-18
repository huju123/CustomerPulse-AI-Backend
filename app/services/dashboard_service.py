import pandas as pd
from pathlib import Path
from app.models.schemas import DashboardSummaryResponse
from app.services import forecast_service

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_CSV = BASE_DIR / "data" / "processed_dataset.csv"


def get_dashboard_summary() -> DashboardSummaryResponse:
    """
    Aggregates real business stats from the processed customer dataset.
    """
    df = pd.read_csv(DATASET_CSV)

    total_customers = len(df)
    churned_customers = int((df["Churn"] == 1).sum())
    active_customers = total_customers - churned_customers
    churn_rate = round((churned_customers / total_customers) * 100, 2)
    total_revenue = round(float(df["total_spend"].sum()), 2)

    # Pull a quick, human-readable line from the real revenue forecast
    forecast_summary = _build_forecast_summary()

    return DashboardSummaryResponse(
        total_customers=total_customers,
        active_customers=active_customers,
        churn_rate=churn_rate,
        total_revenue=total_revenue,
        forecast_summary=forecast_summary,
    )


def _build_forecast_summary() -> str:
    """
    Builds a short, human-readable forecast summary using the real Prophet forecast.
    """
    forecast = forecast_service.get_forecast()
    points = forecast.revenue_forecast

    if not points:
        return "Forecast data is currently unavailable."

    first_value = points[0].predicted_value
    last_value = points[-1].predicted_value

    if last_value > first_value:
        trend = "an upward trend"
    elif last_value < first_value:
        trend = "a downward trend"
    else:
        trend = "a stable trend"

    return (
        f"Revenue is forecasted to show {trend} over the next "
        f"{len(points)} months, based on historical purchase patterns."
    )

