from fastapi import APIRouter, HTTPException
from app.models.schemas import CustomerFeatures, ChurnPredictionResponse
from app.services import churn_service

router = APIRouter(
    prefix="/predict-churn",
    tags=["Churn Prediction"]
)


@router.post("", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerFeatures):
    """
    Predict the churn probability for a given customer using the trained XGBoost model.
    """
    try:
        return churn_service.predict_churn(customer)
    except ValueError as e:
        # e.g. unknown country not seen during training
        raise HTTPException(status_code=400, detail=str(e))