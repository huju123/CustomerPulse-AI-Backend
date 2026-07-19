from fastapi import APIRouter, Query
from app.models.schemas import HighRiskCustomersResponse
from app.services import risk_service

router = APIRouter(
    prefix="/high-risk-customers",
    tags=["Risk Analysis"]
)


@router.get("", response_model=HighRiskCustomersResponse)
def get_high_risk_customers(
    limit: int = Query(default=10, ge=1, le=100, description="Number of top riskiest customers to return")
):
    """
    Scores all customers in the dataset with the real trained churn model,
    returns the top N riskiest customers and a real risk segmentation breakdown.
    """
    return risk_service.get_high_risk_customers(limit=limit)