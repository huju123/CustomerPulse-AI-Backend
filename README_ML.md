# README_ML.md — CustomerPulse AI Backend (ML Integration Reference)

This document explains how the machine learning artifacts handed off by the ML team
were integrated into the FastAPI backend. It's meant as a single reference for
teammates, graders, or anyone reviewing this repository.

---

## 1. Project Overview

**CustomerPulse AI** is a full-stack analytics platform that predicts customer churn,
forecasts business trends, and explains predictions using Explainable AI (SHAP).

This backend (`Customer_Pulse_Backend`) is built with **FastAPI** and exposes four
REST endpoints consumed by the React frontend. It loads pre-trained models and
datasets provided by the ML team rather than training anything itself.

---

## 2. Models Used

| Model | File | Purpose |
|---|---|---|
| XGBoost Classifier | `ml_artifacts/churn_model.pkl` | Predicts churn probability for a customer |
| LabelEncoder (country) | `ml_artifacts/country_encoder.pkl` | Encodes the `country` feature exactly as done during training |
| Prophet | `ml_artifacts/prophet_model.pkl` (forecast precomputed to CSV) | Revenue forecasting |

SHAP (`TreeExplainer`) is applied live, at request time, on top of the loaded
XGBoost model — it is not a separately saved model file.

---

## 3. Input Features (Churn Prediction)

`POST /predict-churn` expects the following 6 features, **in this exact order**,
matching how the model was trained:

```
total_transactions          (int)     — total number of transactions
total_quantity               (int)     — total quantity of items purchased
country                      (string)  — customer's country (encoded internally)
total_spend                  (float)   — total amount spent
days_since_last_purchase     (int)     — days since the customer's last purchase
customer_lifetime_days       (int)     — days between first and last purchase
```

**Country encoding:** `country` is a plain string in the API request
(e.g. `"United Kingdom"`). The backend encodes it internally using the saved
`LabelEncoder` (`country_encoder.pkl`) — the same encoder fit during training.
If an unrecognized country is submitted, the API returns a `400 Bad Request`
listing the valid, known countries rather than silently producing a wrong prediction.

**Feature derivation notes (from the ML team):**
- `customer_lifetime_days = last_purchase_date - first_purchase_date` (in days)
- `country` was encoded using `sklearn.preprocessing.LabelEncoder`

---

## 4. Output Format

### `POST /predict-churn`
```json
{
  "churn_probability": 0.9985,
  "risk_level": "High",
  "confidence_score": 0.9985,
  "top_factors": [
    {
      "feature_name": "days_since_last_purchase",
      "impact": 7.5801,
      "description": "Days since the customer last made a purchase increases churn risk for this customer"
    }
  ],
  "explanation_summary": "This customer has a high risk of churn based on their transaction history, spending, and recency of purchase."
}
```
`top_factors` is computed **live**, per request, using SHAP's `TreeExplainer` on the
loaded XGBoost model — it genuinely reflects the specific customer submitted, not a
static/global value.

### `GET /forecast`
Returns a real Prophet-generated revenue forecast (from `data/revenue_forecast.csv`,
columns `ds`/`yhat`). Negative predicted values are clipped to `0` before returning,
since revenue cannot be negative in reality (a known Prophet artifact on wide
confidence intervals with limited training data).

`customer_growth_forecast`, `active_customers_forecast`, and `churn_trend_forecast`
are currently returned as **empty lists** — no trained model or precomputed data has
been handed off for these three metrics yet. The schema/API already supports them;
they'll be populated once available from the ML team.

### `GET /dashboard-summary`
All four fields (`total_customers`, `active_customers`, `churn_rate`, `total_revenue`)
are computed live from `data/processed_dataset.csv` (4,338 customers) — not
hardcoded. `forecast_summary` is a one-line trend description generated from the
real revenue forecast above.

### `GET /feature-importance`
Returns global SHAP feature importance from `data/shap_feature_importance.csv`
(columns: `Feature`, `Mean SHAP`), sorted by importance. Note: this file currently
includes 5 of the 6 model features — `country` is not listed (flagged to the ML team).

---

## 5. Required Files (must be present for the backend to start)

```
ml_artifacts/
├── churn_model.pkl
├── country_encoder.pkl
└── prophet_model.pkl        (not currently loaded directly — forecast is served from CSV)

data/
├── processed_dataset.csv
├── revenue_forecast.csv
├── shap_values.csv          (per-customer, training set only — not used live)
└── shap_feature_importance.csv
```

Models are loaded **once at application startup** (not per-request) for performance.

---

## 6. API Endpoints

| Method | Endpoint | Status |
|---|---|---|
| POST | `/predict-churn` | ✅ Fully implemented — real XGBoost + live SHAP |
| GET | `/forecast` | ⚠️ Partially implemented — revenue is real; other 3 metrics pending ML team |
| GET | `/dashboard-summary` | ✅ Fully implemented — real aggregated stats |
| GET | `/feature-importance` | ✅ Fully implemented — real global SHAP importance |

---

## 7. Known Limitations / Open Items

- Only revenue forecasting has a real model behind it; customer growth, active
  customers, and churn trend forecasts are not yet available from the ML team.
- `shap_feature_importance.csv` does not currently include the `country` feature.
- `revenue_forecast.csv` does not include Prophet's confidence interval columns
  (`yhat_lower`, `yhat_upper`) — `lower_bound`/`upper_bound` are returned as `null`
  until the ML team re-exports with those columns included.
- Country values submitted to `/predict-churn` must exactly match a country seen
  during training (case-sensitive). The valid list can be checked by loading
  `country_encoder.pkl` and inspecting `.classes_`.

---

## 8. Contact / Ownership

- **Backend & Integration:** Huzaifa
- **ML / Data:** [Teammate name]
