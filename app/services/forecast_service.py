# import pandas as pd
# from pathlib import Path
# from datetime import date
# from app.models.schemas import ForecastResponse, ForecastPoint

# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# REVENUE_FORECAST_CSV = BASE_DIR / "data" / "revenue_forecast.csv"


# def _load_revenue_forecast() -> list[ForecastPoint]:
#     """
#     Load precomputed Prophet revenue forecast.
#     Negative values are clipped to 0 — revenue cannot be negative in reality;
#     this is a known Prophet artifact on wide confidence intervals, not a backend bug.
#     """
#     df = pd.read_csv(REVENUE_FORECAST_CSV)

#     points = []
#     for _, row in df.iterrows():
#         forecast_date = pd.to_datetime(row["ds"]).date()
#         predicted_value = max(float(row["yhat"]), 0.0)  # clip negative revenue to 0

#         points.append(
#             ForecastPoint(
#                 date=forecast_date,
#                 predicted_value=round(predicted_value, 2),
#                 lower_bound=None,
#                 upper_bound=None,
#             )
#         )
#     return points


# def get_forecast() -> ForecastResponse:
#     """
#     Returns business forecasts.
#     Revenue forecast: real Prophet model output.
#     Customer growth, active customers, churn trend: synthetic data pending ML model development.
#     """
#     revenue_points = _load_revenue_forecast()
#     num_points = len(revenue_points)
    
#     # Generate synthetic customer growth forecast (trending up with noise)
#     customer_growth = [
#         ForecastPoint(
#             date=revenue_points[i].date,
#             predicted_value=round(50 + (i * 8) + (i % 3) * 12, 2),
#             lower_bound=None,
#             upper_bound=None,
#         )
#         for i in range(num_points)
#     ]
    
#     # Generate synthetic active customers forecast (steady growth)
#     active_customers = [
#         ForecastPoint(
#             date=revenue_points[i].date,
#             predicted_value=round(200 + (i * 15) + (i % 4) * 20, 2),
#             lower_bound=None,
#             upper_bound=None,
#         )
#         for i in range(num_points)
#     ]
    
#     # Generate synthetic churn trend forecast (declining churn rate)
#     churn_trend = [
#         ForecastPoint(
#             date=revenue_points[i].date,
#             predicted_value=round(max(8.5 - (i * 0.4), 1.0), 2),
#             lower_bound=None,
#             upper_bound=None,
#         )
#         for i in range(num_points)
#     ]
    
#     return ForecastResponse(
#         revenue_forecast=revenue_points,
#         customer_growth_forecast=customer_growth,
#         active_customers_forecast=active_customers,
#         churn_trend_forecast=churn_trend,
#         forecast_generated_on=date.today(),
#         forecast_horizon_days=len(revenue_points) * 30,
#     )

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from app.models.schemas import ForecastResponse, ForecastPoint

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REVENUE_FORECAST_CSV = BASE_DIR / "data" / "revenue_forecast.csv"
PROCESSED_DATASET_CSV = BASE_DIR / "data" / "processed_dataset.csv"


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
        predicted_value = max(float(row["yhat"]), 0.0)

        points.append(
            ForecastPoint(
                date=forecast_date,
                predicted_value=round(predicted_value, 2),
                lower_bound=None,
                upper_bound=None,
            )
        )
    return points


def _linear_trend_forecast(monthly_series: pd.Series, future_dates: list) -> list[float]:
    """
    Fits a simple linear trend (least-squares) to a real historical monthly series
    and projects it forward for the given future dates.
    This is a lightweight, transparent statistical projection — NOT a claim of
    Prophet-level accuracy. It is clearly documented as trend-based estimation.
    """
    if len(monthly_series) < 2:
        # Not enough history to fit a trend — fall back to flat projection of the last known value
        last_value = float(monthly_series.iloc[-1]) if len(monthly_series) else 0.0
        return [round(last_value, 2) for _ in future_dates]

    x = np.arange(len(monthly_series))
    y = monthly_series.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(monthly_series), len(monthly_series) + len(future_dates))
    projected = slope * future_x + intercept

    # Values like customer counts / rates cannot go negative
    return [round(max(float(v), 0.0), 2) for v in projected]


def _load_customer_growth_and_active_and_churn(
    future_dates: list,
) -> tuple[list[ForecastPoint], list[ForecastPoint], list[ForecastPoint]]:
    """
    Derives Customer Growth, Active Customers, and Churn Trend forecasts from
    REAL historical data (first_purchase_date, last_purchase_date, Churn) in
    processed_dataset.csv, using a simple linear trend projection.

    This is an interim approach until the ML team trains dedicated forecasting
    models for these three metrics (only Revenue currently has one).
    """
    df = pd.read_csv(PROCESSED_DATASET_CSV)
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

    # --- Customer growth: new customers acquired per historical month ---
    monthly_new_customers = (
        df.set_index("first_purchase_date").resample("ME").size().sort_index()
    )
    growth_values = _linear_trend_forecast(monthly_new_customers, future_dates)
    customer_growth = [
        ForecastPoint(date=d, predicted_value=v, lower_bound=None, upper_bound=None)
        for d, v in zip(future_dates, growth_values)
    ]

    # --- Active customers: cumulative customers acquired, month over month ---
    cumulative_active = monthly_new_customers.cumsum()
    active_values = _linear_trend_forecast(cumulative_active, future_dates)
    # Cumulative counts should never dip below the last known real total
    last_known_total = float(cumulative_active.iloc[-1]) if len(cumulative_active) else 0.0
    active_values = [round(max(v, last_known_total), 2) for v in active_values]
    active_customers = [
        ForecastPoint(date=d, predicted_value=v, lower_bound=None, upper_bound=None)
        for d, v in zip(future_dates, active_values)
    ]

    # --- Churn trend: monthly churn rate among customers whose last purchase falls in that month ---
    monthly_churn_rate = (
        df.set_index("last_purchase_date")
        .resample("ME")["Churn"]
        .mean()
        .mul(100)
        .sort_index()
    )
    churn_values = _linear_trend_forecast(monthly_churn_rate, future_dates)
    # Churn rate is a percentage, cap at 100
    churn_values = [round(min(v, 100.0), 2) for v in churn_values]
    churn_trend = [
        ForecastPoint(date=d, predicted_value=v, lower_bound=None, upper_bound=None)
        for d, v in zip(future_dates, churn_values)
    ]

    return customer_growth, active_customers, churn_trend


def get_forecast() -> ForecastResponse:
    """
    Returns business forecasts.
    - Revenue: real Prophet model output (data/revenue_forecast.csv).
    - Customer growth, active customers, churn trend: derived from REAL historical
      data via linear trend projection (interim method, pending dedicated
      Prophet models per metric from the ML team).
    """
    revenue_points = _load_revenue_forecast()
    future_dates = [p.date for p in revenue_points]

    customer_growth, active_customers, churn_trend = _load_customer_growth_and_active_and_churn(
        future_dates
    )

    return ForecastResponse(
        revenue_forecast=revenue_points,
        customer_growth_forecast=customer_growth,
        active_customers_forecast=active_customers,
        churn_trend_forecast=churn_trend,
        forecast_generated_on=date.today(),
        forecast_horizon_days=len(revenue_points) * 30,
    )

