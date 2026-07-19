import joblib
import pandas as pd
from pathlib import Path
from app.models.schemas import HighRiskCustomer, RiskSegment, HighRiskCustomersResponse, RiskLevel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "ml_artifacts" / "churn_model.pkl"
ENCODER_PATH = BASE_DIR / "ml_artifacts" / "country_encoder.pkl"
PROCESSED_DATASET_CSV = BASE_DIR / "data" / "processed_dataset.csv"

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
    if prob >= 0.66:
        return RiskLevel.HIGH
    elif prob >= 0.33:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the exact feature set the model expects, for the WHOLE dataset at once
    (batch scoring — much faster than looping and calling the model per row).
    """
    df = df.copy()
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
    df["customer_lifetime_days"] = (df["last_purchase_date"] - df["first_purchase_date"]).dt.days

    # Encode country; unseen/unknown countries are dropped rather than guessed
    known_countries = set(_country_encoder.classes_)
    df = df[df["country"].isin(known_countries)].copy()
    df["country_encoded"] = _country_encoder.transform(df["country"])

    features = df[[
        "total_transactions",
        "total_quantity",
        "total_spend",
        "days_since_last_purchase",
        "customer_lifetime_days",
    ]].copy()
    features["country"] = df["country_encoded"]
    features = features[FEATURE_ORDER]

    return df, features


def get_high_risk_customers(limit: int = 10) -> HighRiskCustomersResponse:
    """
    Scores every customer in the dataset using the real trained XGBoost model,
    returns the top `limit` riskiest customers, plus a real risk segmentation
    breakdown (Low/Medium/High counts and percentages) across the FULL dataset.
    """
    raw_df = pd.read_csv(PROCESSED_DATASET_CSV)
    df, features = _prepare_features(raw_df)

    churn_probabilities = _model.predict_proba(features)[:, 1]
    df["churn_probability"] = churn_probabilities
    df["risk_level"] = df["churn_probability"].apply(_risk_level_from_probability)

    total_scored = len(df)

    # --- Top N riskiest customers ---
    top_df = df.sort_values("churn_probability", ascending=False).head(limit)
    customers = [
        HighRiskCustomer(
            customer_id=str(int(row["CustomerID"])) if pd.notna(row["CustomerID"]) else "Unknown",
            churn_probability=round(float(row["churn_probability"]), 4),
            risk_level=row["risk_level"],
            total_spend=round(float(row["total_spend"]), 2),
            days_since_last_purchase=int(row["days_since_last_purchase"]),
        )
        for _, row in top_df.iterrows()
    ]

    # --- Real risk segmentation across the FULL scored dataset ---
    # NOTE: pandas' vectorized `==` on a column of str-based Enum values (RiskLevel)
    # does not compare correctly (a known pandas/numpy quirk with str+Enum hybrids).
    # Using .apply() with a plain lambda avoids it and gives correct counts.
    segmentation = []
    for level in [RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
        count = int(df["risk_level"].apply(lambda x, lvl=level: x == lvl).sum())
        percentage = round((count / total_scored) * 100, 2) if total_scored else 0.0
        segmentation.append(
            RiskSegment(risk_level=level, customer_count=count, percentage=percentage)
        )

    return HighRiskCustomersResponse(
        customers=customers,
        risk_segmentation=segmentation,
        total_customers_scored=total_scored,
    )