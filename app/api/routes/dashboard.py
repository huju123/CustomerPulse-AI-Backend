from fastapi import APIRouter
from app.models.schemas import DashboardSummaryResponse

router = APIRouter(
    prefix="/dashboard-summary",
    tags=["Dashboard"]
)


@router.get("", response_model=DashboardSummaryResponse)
def get_dashboard_summary():
    """
    Returns high-level dashboard overview stats.
    Currently returns dummy data — will be wired to dashboard_service.py (aggregating real data).
    """
    # TODO: Replace with real aggregation logic via dashboard_service.py
    return DashboardSummaryResponse(
        total_customers=5000,
        active_customers=4200,
        churn_rate=16.0,
        total_revenue=285000.50,
        forecast_summary="Revenue is projected to grow 8% over the next quarter."
    )