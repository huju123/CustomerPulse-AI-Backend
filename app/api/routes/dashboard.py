from fastapi import APIRouter
from app.models.schemas import DashboardSummaryResponse
from app.services import dashboard_service

router = APIRouter(
    prefix="/dashboard-summary",
    tags=["Dashboard"]
)


@router.get("", response_model=DashboardSummaryResponse)
def get_dashboard_summary():
    """
    Returns real aggregate business stats computed from the processed customer dataset.
    """
    return dashboard_service.get_dashboard_summary()