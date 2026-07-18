from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import churn, forecast, dashboard, feature_importance

app = FastAPI(
    title="CustomerPulse AI - Backend",
    description="Predictive Customer Churn & Business Forecasting API",
    version="1.0.0"
)

# ============================================================
# CORS Configuration
# ============================================================
# Allows the React frontend (running on a different port/origin)
# to make requests to this API from the browser.
origins = [
    "http://localhost:5173",   # Vite dev server default port
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],   # allow all headers (Content-Type, Authorization, etc.)
)

# Register all route modules
app.include_router(churn.router)
app.include_router(forecast.router)
app.include_router(dashboard.router)
app.include_router(feature_importance.router)


@app.get("/", tags=["Root"])
def root():
    return {"message": "CustomerPulse AI Backend is running 🚀"}