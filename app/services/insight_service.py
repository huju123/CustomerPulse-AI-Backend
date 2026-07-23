from app.models.schemas import BusinessInsightsResponse, Recommendation
from app.services import dashboard_service, risk_service, shap_service, forecast_service


def get_business_insights() -> BusinessInsightsResponse:
    """
    Generates business recommendations and next-best actions using simple,
    transparent RULES applied to real, already-computed data:
    - dashboard_service (churn rate, revenue)
    - risk_service (high-risk customer segmentation)
    - shap_service (top churn-driving feature)
    - forecast_service (revenue trend direction)

    This is rule-based, not AI-generated text — each recommendation is derived
    from a specific real number, and the logic is intentionally readable so it
    can be explained in a demo or report.
    """
    summary = dashboard_service.get_dashboard_summary()
    risk_data = risk_service.get_high_risk_customers(limit=1)
    top_features = shap_service.get_global_feature_importance()
    forecast = forecast_service.get_forecast()

    recommendations = []
    actions = []

    # --- Rule 1: churn rate severity ---
    churn_rate = summary.churn_rate
    if churn_rate >= 20:
        recommendations.append(
            Recommendation(
                title="Urgent Retention Intervention",
                description=(
                    f"Churn rate is {churn_rate}%, which is high. Prioritize proactive "
                    f"outreach to at-risk customers before their next renewal window."
                ),
            )
        )
        actions.append(f"Churn rate is {churn_rate}% — launch a targeted retention campaign this week.")
    elif churn_rate >= 10:
        recommendations.append(
            Recommendation(
                title="Monitor Churn Trend Closely",
                description=(
                    f"Churn rate is {churn_rate}%, a moderate level. Track it weekly and "
                    f"identify which segments are driving it."
                ),
            )
        )
        actions.append(f"Churn rate is {churn_rate}% — review weekly to catch early increases.")
    else:
        recommendations.append(
            Recommendation(
                title="Maintain Current Retention Practices",
                description=f"Churn rate is {churn_rate}%, a healthy level. Continue current retention efforts.",
            )
        )
        actions.append(f"Churn rate is stable at {churn_rate}% — maintain current retention practices.")

    # --- Rule 2: top SHAP-driven feature across the model ---
    if top_features:
        top_feature = max(top_features, key=lambda f: f.importance_score)
        readable_names = {
            "days_since_last_purchase": "time since a customer's last purchase",
            "customer_lifetime_days": "how long a customer has been active",
            "total_quantity": "total quantity of items purchased",
            "total_transactions": "number of past transactions",
            "total_spend": "total amount spent",
            "country": "customer's country",
        }
        readable = readable_names.get(top_feature.feature_name, top_feature.feature_name)
        recommendations.append(
            Recommendation(
                title="Target the Strongest Churn Driver",
                description=(
                    f"'{top_feature.feature_name}' ({readable}) is the strongest churn predictor "
                    f"across the customer base. Design retention triggers around this signal, "
                    f"e.g. automatic outreach once a customer crosses a risk threshold on it."
                ),
            )
        )
        actions.append(f"Build an automated alert when a customer's '{top_feature.feature_name}' crosses a risk threshold.")

    # --- Rule 3: high-risk customer volume ---
    high_segment = next((s for s in risk_data.risk_segmentation if s.risk_level == "High"), None)
    if high_segment and high_segment.customer_count > 0:
        recommendations.append(
            Recommendation(
                title="Prioritize High-Risk Segment",
                description=(
                    f"{high_segment.customer_count} customers ({high_segment.percentage}%) are "
                    f"currently classified as High risk. Focus retention budget on this group first, "
                    f"since they represent the highest-probability churners."
                ),
            )
        )
        actions.append(
            f"Review the {high_segment.customer_count} High-risk customers in Risk Analysis and assign follow-ups."
        )

    # --- Rule 4: revenue forecast direction ---
    if forecast.revenue_forecast:
        first_value = forecast.revenue_forecast[0].predicted_value
        last_value = forecast.revenue_forecast[-1].predicted_value
        if last_value < first_value:
            actions.append("Revenue forecast trends downward over the horizon — revisit pricing or upsell strategy.")
        elif last_value > first_value:
            actions.append("Revenue forecast trends upward — maintain current growth initiatives.")

    return BusinessInsightsResponse(
        recommendations=recommendations[:3],
        actions=actions[:4],
    )