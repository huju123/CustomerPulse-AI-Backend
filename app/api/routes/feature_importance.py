from fastapi import APIRouter
from app.models.schemas import FeatureImportanceResponse, FeatureImportanceItem

router = APIRouter(
    prefix="/feature-importance",
    tags=["Explainability"]
)


@router.get("", response_model=FeatureImportanceResponse)
def get_feature_importance():
    """
    Returns global feature importance from the trained churn model.
    Currently returns dummy data — will be wired to shap_service.py.
    """
    # TODO: Replace with real SHAP-based global importance via shap_service.py
    return FeatureImportanceResponse(
        model_name="XGBoost",
        features=[
            FeatureImportanceItem(feature_name="contract_type", importance_score=0.28),
            FeatureImportanceItem(feature_name="tenure_months", importance_score=0.22),
            FeatureImportanceItem(feature_name="monthly_charges", importance_score=0.18),
            FeatureImportanceItem(feature_name="num_support_tickets", importance_score=0.12),
            FeatureImportanceItem(feature_name="has_tech_support", importance_score=0.10),
            FeatureImportanceItem(feature_name="has_internet_service", importance_score=0.10),
        ]
    )