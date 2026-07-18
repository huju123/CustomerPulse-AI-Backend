import joblib
import pandas as pd
from pathlib import Path
from app.models.schemas import CustomerFeatures, ChurnPredictionResponse, RiskLevel
from app.services import shap_service

# ============================================================
# Load model + encoder ONCE at startup (not per-request)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
MODEL_PATH = BASE_DIR / "ml_artifacts" / "churn_model.pkl"
ENCODER_PATH = BASE_DIR / "ml_artifacts" / "country_encoder.pkl"

_model = joblib.load(MODEL_PATH)
_country_encoder = joblib.load(ENCODER_PATH)

FEATURE_ORDER = [
    "total_transactions",
    "total_quantity",
    "country",
    "total_spend",
    "days_since_last_purchase",
    "customer_lifetime_days",
]


def _risk_level_from_probability(prob: float) -> RiskLevel:
    """Convert a raw probability into a business-friendly risk label."""
    if prob >= 0.66:
        return RiskLevel.HIGH
    elif prob >= 0.33:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _encode_country(country: str) -> int:
    """
    Encode a country string using the SAME LabelEncoder used during training.
    Raises a clear error if the country wasn't seen during training.
    """
    try:
        return int(_country_encoder.transform([country])[0])
    except ValueError:
        known_countries = list(_country_encoder.classes_)
        raise ValueError(
            f"Unknown country '{country}'. Model was trained on: {known_countries}"
        )


def predict_churn(customer: CustomerFeatures) -> ChurnPredictionResponse:
    """
    Run real churn prediction using the trained XGBoost model.
    """
    encoded_country = _encode_country(customer.country)

    # Build the feature row in the EXACT order the model expects
    input_df = pd.DataFrame([{
        "total_transactions": customer.total_transactions,
        "total_quantity": customer.total_quantity,
        "country": encoded_country,
        "total_spend": customer.total_spend,
        "days_since_last_purchase": customer.days_since_last_purchase,
        "customer_lifetime_days": customer.customer_lifetime_days,
    }])[FEATURE_ORDER]

    # Run prediction — predict_proba returns [[prob_class_0, prob_class_1]]
    churn_probability = float(_model.predict_proba(input_df)[0][1])
    risk_level = _risk_level_from_probability(churn_probability)
    confidence_score = round(max(churn_probability, 1 - churn_probability), 4)
    top_factors = shap_service.get_top_factors_for_customer(input_df)

    return ChurnPredictionResponse(
        churn_probability=round(churn_probability, 4),
        risk_level=risk_level,
        confidence_score=confidence_score,
        top_factors=top_factors,
        explanation_summary=(
            f"This customer has a {risk_level.value.lower()} risk of churn "
            f"based on their transaction history, spending, and recency of purchase."
        )
    )