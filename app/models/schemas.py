# from pydantic import BaseModel, Field, ConfigDict
# from typing import List, Optional
# from enum import Enum
# from datetime import date


# # ============================================================
# # ENUMS
# # ============================================================

# class RiskLevel(str, Enum):
#     """Customer churn risk classification levels."""
#     LOW = "Low"
#     MEDIUM = "Medium"
#     HIGH = "High"


# class ContractType(str, Enum):
#     """Customer contract type (adjust based on actual dataset)."""
#     MONTH_TO_MONTH = "Month-to-month"
#     ONE_YEAR = "One year"
#     TWO_YEAR = "Two year"


# # ============================================================
# # 1. POST /predict-churn
# # ============================================================

# class CustomerFeatures(BaseModel):
#     """
#     Request schema for churn prediction.
#     Fields MUST match the exact order/names the trained XGBoost model expects.
#     Source: ML Team handover (churn_model.pkl)
#     """
#     model_config = ConfigDict(
#         json_schema_extra={
#             "example": {
#                 "total_transactions": 12,
#                 "total_quantity": 340,
#                 "country": "United Kingdom",
#                 "total_spend": 1580.50,
#                 "days_since_last_purchase": 45,
#                 "customer_lifetime_days": 365
#             }
#         }
#     )

#     total_transactions: int = Field(..., ge=0, description="Total number of transactions made by the customer")
#     total_quantity: int = Field(..., ge=0, description="Total quantity of items purchased")
#     country: str = Field(..., description="Customer's country (will be label-encoded internally)")
#     total_spend: float = Field(..., ge=0, description="Total amount spent by the customer")
#     days_since_last_purchase: int = Field(..., ge=0, description="Days since the customer's last purchase")
#     customer_lifetime_days: int = Field(..., ge=0, description="Days between first and last purchase")

# class TopFeatureContribution(BaseModel):
#     """A single feature's contribution to the prediction (from SHAP)."""
#     feature_name: str
#     impact: float = Field(..., description="SHAP value — positive pushes toward churn, negative away from it")
#     description: Optional[str] = Field(None, description="Human-readable explanation of this factor")


# class ChurnPredictionResponse(BaseModel):
#     """Response schema for churn prediction."""
#     churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn, between 0 and 1")
#     risk_level: RiskLevel
#     confidence_score: float = Field(..., ge=0, le=1, description="Model's confidence in this prediction")
#     top_factors: List[TopFeatureContribution] = Field(
#         default_factory=list,
#         description="Top SHAP-driven factors influencing this prediction"
#     )
#     explanation_summary: str = Field(..., description="Plain-language explanation of the prediction")


# # ============================================================
# # 2. GET /forecast
# # ============================================================

# class ForecastPoint(BaseModel):
#     """A single forecasted data point on a given date."""
#     date: date
#     predicted_value: float
#     lower_bound: Optional[float] = Field(None, description="Lower confidence interval bound")
#     upper_bound: Optional[float] = Field(None, description="Upper confidence interval bound")


# class ForecastResponse(BaseModel):
#     """Response schema for business forecasting."""
#     revenue_forecast: List[ForecastPoint]
#     customer_growth_forecast: List[ForecastPoint]
#     active_customers_forecast: List[ForecastPoint]
#     churn_trend_forecast: List[ForecastPoint]
#     forecast_generated_on: date
#     forecast_horizon_days: int = Field(..., description="Number of days into the future this forecast covers")


# # ============================================================
# # 3. GET /dashboard-summary
# # ============================================================

# class DashboardSummaryResponse(BaseModel):
#     """Response schema for the dashboard overview page."""
#     total_customers: int = Field(..., ge=0)
#     active_customers: int = Field(..., ge=0)
#     churn_rate: float = Field(..., ge=0, le=100, description="Churn rate as a percentage")
#     total_revenue: float = Field(..., ge=0)
#     forecast_summary: str = Field(..., description="Short human-readable forecast summary for the dashboard")


# # ============================================================
# # 4. GET /feature-importance
# # ============================================================

# class FeatureImportanceItem(BaseModel):
#     """A single feature and its overall importance score."""
#     feature_name: str
#     importance_score: float = Field(..., ge=0, description="Relative importance of this feature in the model")


# class FeatureImportanceResponse(BaseModel):
#     """Response schema for global feature importance (model-level, not per-prediction)."""
#     model_name: str = Field(..., description="Which model this importance data comes from (e.g. XGBoost)")
#     features: List[FeatureImportanceItem]


# # ============================================================
# # 5. GET /high-risk-customers
# # ============================================================

# class HighRiskCustomer(BaseModel):
#     """A single customer's churn risk, computed from the real trained model."""
#     customer_id: str
#     churn_probability: float = Field(..., ge=0, le=1)
#     risk_level: RiskLevel
#     total_spend: float = Field(..., ge=0)
#     days_since_last_purchase: int = Field(..., ge=0)


# class RiskSegment(BaseModel):
#     """Count and percentage of customers in a given risk band, across the FULL dataset."""
#     risk_level: RiskLevel
#     customer_count: int = Field(..., ge=0)
#     percentage: float = Field(..., ge=0, le=100)


# class HighRiskCustomersResponse(BaseModel):
#     """Response schema for the high-risk customer list + full-dataset risk segmentation."""
#     customers: List[HighRiskCustomer] = Field(..., description="Top N riskiest customers, sorted by churn probability descending")
#     risk_segmentation: List[RiskSegment] = Field(..., description="Risk band breakdown across ALL customers in the dataset")
#     total_customers_scored: int = Field(..., ge=0)

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from enum import Enum
from datetime import date


# ============================================================
# ENUMS
# ============================================================

class RiskLevel(str, Enum):
    """Customer churn risk classification levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ContractType(str, Enum):
    """Customer contract type (adjust based on actual dataset)."""
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


# ============================================================
# 1. POST /predict-churn
# ============================================================

class CustomerFeatures(BaseModel):
    """
    Request schema for churn prediction.
    Fields MUST match the exact order/names the trained XGBoost model expects.
    Source: ML Team handover (churn_model.pkl)
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_transactions": 12,
                "total_quantity": 340,
                "country": "United Kingdom",
                "total_spend": 1580.50,
                "days_since_last_purchase": 45,
                "customer_lifetime_days": 365
            }
        }
    )

    total_transactions: int = Field(..., ge=0, description="Total number of transactions made by the customer")
    total_quantity: int = Field(..., ge=0, description="Total quantity of items purchased")
    country: str = Field(..., description="Customer's country (will be label-encoded internally)")
    total_spend: float = Field(..., ge=0, description="Total amount spent by the customer")
    days_since_last_purchase: int = Field(..., ge=0, description="Days since the customer's last purchase")
    customer_lifetime_days: int = Field(..., ge=0, description="Days between first and last purchase")

class TopFeatureContribution(BaseModel):
    """A single feature's contribution to the prediction (from SHAP)."""
    feature_name: str
    impact: float = Field(..., description="SHAP value — positive pushes toward churn, negative away from it")
    description: Optional[str] = Field(None, description="Human-readable explanation of this factor")


class ChurnPredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn, between 0 and 1")
    risk_level: RiskLevel
    confidence_score: float = Field(..., ge=0, le=1, description="Model's confidence in this prediction")
    top_factors: List[TopFeatureContribution] = Field(
        default_factory=list,
        description="Top SHAP-driven factors influencing this prediction"
    )
    explanation_summary: str = Field(..., description="Plain-language explanation of the prediction")


# ============================================================
# 2. GET /forecast
# ============================================================

class ForecastPoint(BaseModel):
    """A single forecasted data point on a given date."""
    date: date
    predicted_value: float
    lower_bound: Optional[float] = Field(None, description="Lower confidence interval bound")
    upper_bound: Optional[float] = Field(None, description="Upper confidence interval bound")


class ForecastResponse(BaseModel):
    """Response schema for business forecasting."""
    revenue_forecast: List[ForecastPoint]
    customer_growth_forecast: List[ForecastPoint]
    active_customers_forecast: List[ForecastPoint]
    churn_trend_forecast: List[ForecastPoint]
    forecast_generated_on: date
    forecast_horizon_days: int = Field(..., description="Number of days into the future this forecast covers")


# ============================================================
# 3. GET /dashboard-summary
# ============================================================

class DashboardSummaryResponse(BaseModel):
    """Response schema for the dashboard overview page."""
    total_customers: int = Field(..., ge=0)
    active_customers: int = Field(..., ge=0)
    churn_rate: float = Field(..., ge=0, le=100, description="Churn rate as a percentage")
    total_revenue: float = Field(..., ge=0)
    forecast_summary: str = Field(..., description="Short human-readable forecast summary for the dashboard")


# ============================================================
# 4. GET /feature-importance
# ============================================================

class FeatureImportanceItem(BaseModel):
    """A single feature and its overall importance score."""
    feature_name: str
    importance_score: float = Field(..., ge=0, description="Relative importance of this feature in the model")


class FeatureImportanceResponse(BaseModel):
    """Response schema for global feature importance (model-level, not per-prediction)."""
    model_name: str = Field(..., description="Which model this importance data comes from (e.g. XGBoost)")
    features: List[FeatureImportanceItem]


# ============================================================
# 5. GET /high-risk-customers
# ============================================================

class HighRiskCustomer(BaseModel):
    """A single customer's churn risk, computed from the real trained model."""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    total_spend: float = Field(..., ge=0)
    days_since_last_purchase: int = Field(..., ge=0)


class RiskSegment(BaseModel):
    """Count and percentage of customers in a given risk band, across the FULL dataset."""
    risk_level: RiskLevel
    customer_count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0, le=100)


class HighRiskCustomersResponse(BaseModel):
    """Response schema for the high-risk customer list + full-dataset risk segmentation."""
    customers: List[HighRiskCustomer] = Field(..., description="Top N riskiest customers, sorted by churn probability descending")
    risk_segmentation: List[RiskSegment] = Field(..., description="Risk band breakdown across ALL customers in the dataset")
    total_customers_scored: int = Field(..., ge=0)


# ============================================================
# 6. GET /business-insights
# ============================================================

class Recommendation(BaseModel):
    """A single retention recommendation, generated from real data via rule-based logic."""
    title: str
    description: str


class BusinessInsightsResponse(BaseModel):
    """Response schema for rule-based business recommendations and next-best actions."""
    recommendations: List[Recommendation]
    actions: List[str]