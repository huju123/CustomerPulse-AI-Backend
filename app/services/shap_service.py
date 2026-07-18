import shap
import joblib
import pandas as pd
from pathlib import Path
from app.models.schemas import TopFeatureContribution, FeatureImportanceItem

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "ml_artifacts" / "churn_model.pkl"
FEATURE_IMPORTANCE_CSV = BASE_DIR / "data" / "shap_feature_importance.csv"

_model = joblib.load(MODEL_PATH)

# TreeExplainer is fast and built specifically for tree-based models like XGBoost
_explainer = shap.TreeExplainer(_model)

# Human-readable descriptions for each feature (used in per-customer explanations)
FEATURE_DESCRIPTIONS = {
    "total_transactions": "Number of past transactions",
    "total_quantity": "Total quantity of items purchased",
    "country": "Customer's country",
    "total_spend": "Total amount spent",
    "days_since_last_purchase": "Days since the customer last made a purchase",
    "customer_lifetime_days": "How long the customer has been active",
}


def get_top_factors_for_customer(input_df: pd.DataFrame, top_n: int = 3) -> list[TopFeatureContribution]:
    """
    Compute REAL per-customer SHAP values for a single prediction input.
    input_df must already have features in the correct model order (single row).
    """
    shap_values = _explainer.shap_values(input_df)

    # shap_values shape: (1, num_features) for a single row
    row_shap = shap_values[0]
    feature_names = input_df.columns.tolist()

    # Pair each feature with its SHAP impact, sort by absolute impact (highest first)
    contributions = list(zip(feature_names, row_shap))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = []
    for feature_name, impact in contributions[:top_n]:
        direction = "increases" if impact > 0 else "decreases"
        top_factors.append(
            TopFeatureContribution(
                feature_name=feature_name,
                impact=round(float(impact), 4),
                description=f"{FEATURE_DESCRIPTIONS.get(feature_name, feature_name)} {direction} churn risk for this customer"
            )
        )

    return top_factors


def get_global_feature_importance() -> list[FeatureImportanceItem]:
    """
    Load precomputed global feature importance (from the ML team's shap_feature_importance.csv).
    """
    df = pd.read_csv(FEATURE_IMPORTANCE_CSV)

    return [
        FeatureImportanceItem(
            feature_name=row["Feature"],
            importance_score=round(float(row["Mean SHAP"]), 4)
        )
        for _, row in df.iterrows()
    ]

