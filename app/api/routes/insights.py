from fastapi import APIRouter
from app.models.schemas import BusinessInsightsResponse
from app.services import insights_service

router = APIRouter(
    prefix="/business-insights",
    tags=["Business Insights"]
)


@router.get("", response_model=BusinessInsightsResponse)
def get_business_insights():
    """
    Returns rule-based recommendations and next-best actions, derived from
    real dashboard, risk, SHAP, and forecast data.
    """
    return insights_service.get_business_insights()