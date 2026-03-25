# uvicorn src.api:app --reload --port 8000
# http://127.0.0.1:8000/docs --> Open In Chrome using Swagger UI

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import shap
from datetime import datetime

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from src.drift import MovingAverageDriftDetector
from src.drift_adaptive import ADWINDriftDetector, DDMDriftDetector
from src.rollback import ModelManager
from src.drift_logger import DriftLogger
from src.shap_logger import ShapLogger

from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# App
# =========================================================
app = FastAPI(
    title="Trustworthy NWDAF ML-Ops API",
    version="3.1.0",
    description=(
        "Federated Learning inference with hybrid drift detection, "
        "rollback, auto-recovery, and SHAP-based explainability"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================================================
# Load FL Global Model
# =========================================================
FL_MODEL_PATH = "models/fl_global_model.pkl"
MODEL_VERSION = "FL-v1.0"

fl_model_data = joblib.load(FL_MODEL_PATH)

fl_model = SGDRegressor()
fl_model.coef_ = np.array(fl_model_data["coef"])
fl_model.intercept_ = np.array(fl_model_data["intercept"])

model_manager = ModelManager(
    stable_model=fl_model,
    primary_model=fl_model
)

# =========================================================
# Drift Detectors (HYBRID STRATEGY)
# =========================================================
ma_detector = MovingAverageDriftDetector(
    window_size=50,
    threshold=0.3,
    min_drift_windows=3
)

adwin_detector = ADWINDriftDetector(delta=0.002)
ddm_detector = DDMDriftDetector()

PRIMARY_DETECTOR_NAME = "MovingAverage"
drift_logger = DriftLogger("drift_events.csv")

# =========================================================
# SHAP Explainer (GLOBAL, SAFE)
# =========================================================
background = np.zeros((50, 6))
shap_masker = shap.maskers.Independent(background)

shap_explainer = shap.LinearExplainer(
    model_manager.primary_model,
    shap_masker
)

shap_logger = ShapLogger()

SHAP_FEATURE_NAMES = [
    "time_of_day",
    "slice_type",
    "jitter",
    "packet_loss",
    "throughput"
]

# =========================================================
# Input / Output Schemas
# =========================================================
class TrafficSample(BaseModel):
    time_of_day: int = Field(..., ge=0, le=23)
    slice_type: str = Field(..., description="eMBB / URLLC / mMTC")
    jitter: float = Field(..., ge=0)
    packet_loss: float = Field(..., ge=0, le=100)
    throughput: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    predicted_future_load: float
    explanation: str
    model_version: str
    warning: str | None = None


class ExplainResponse(BaseModel):
    prediction: float
    shap_values: dict
    note: str


# =========================================================
# Utilities
# =========================================================
SLICE_MAP = {"eMBB": 0, "URLLC": 1, "mMTC": 2}

scaler = StandardScaler()  # used ONLY for consistency, not re-fit


def generate_explanation(sample: TrafficSample) -> str:
    reasons = []

    if sample.jitter > 30:
        reasons.append("high jitter")
    if sample.packet_loss > 2:
        reasons.append("packet loss")
    if sample.throughput > 800:
        reasons.append("heavy throughput usage")

    if reasons:
        return f"Predicted load may increase due to {', '.join(reasons)}."
    return "Network conditions appear stable with moderate predicted load."


def build_feature_vector(sample: TrafficSample) -> np.ndarray:
    slice_encoded = SLICE_MAP.get(sample.slice_type, 0)

    return np.array([[
        sample.time_of_day,
        slice_encoded,
        sample.jitter,
        sample.packet_loss,
        sample.throughput,
        0.0  # placeholder (matches FL training)
    ]])


# =========================================================
# Prediction Endpoint
# =========================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(sample: TrafficSample):

    X = build_feature_vector(sample)

    # SLA metric for drift detection
    sla_metric = sample.jitter + sample.packet_loss

    # Drift signals
    ma_drift = ma_detector.update(sla_metric)
    adwin_drift = adwin_detector.update(sla_metric)
    ddm_drift = ddm_detector.update(sla_metric)

    recovered = False

    # ---- Rollback ONLY on MA drift ----
    if ma_drift and not model_manager.rollback_active:
        model_manager.rollback()

        drift_logger.log(
            detector_name=PRIMARY_DETECTOR_NAME,
            sla_metric=sla_metric,
            model_version=MODEL_VERSION,
            action="rollback"
        )

    # ---- Auto-recovery ----
    recovered = model_manager.try_recover(system_stable=not ma_drift)

    if recovered:
        drift_logger.log(
            detector_name=PRIMARY_DETECTOR_NAME,
            sla_metric=sla_metric,
            model_version=MODEL_VERSION,
            action="recovered"
        )

    # ---- Prediction ----
    prediction = model_manager.predict(X)[0]

    # ---- Warning message ----
    warning = None
    if model_manager.rollback_active:
        warning = "⚠ Concept drift detected. Rolled back to stable model."
    elif recovered:
        warning = "✅ System stabilized. Primary model restored."
    elif adwin_drift or ddm_drift:
        warning = "ℹ Early drift warning detected (adaptive detector)."

    return PredictionResponse(
        predicted_future_load=float(prediction),
        explanation=generate_explanation(sample),
        model_version=f"{MODEL_VERSION} ({model_manager.status()})",
        warning=warning
    )


# =========================================================
# Explain Endpoint (SHAP)
# =========================================================
@app.post("/explain", response_model=ExplainResponse)
def explain(sample: TrafficSample):

    X = build_feature_vector(sample)

    prediction = model_manager.predict(X)[0]

    shap_values_full = shap_explainer.shap_values(X)
    shap_values = shap_values_full[0][:-1]  # drop placeholder

    shap_dict = {
        name: float(val)
        for name, val in zip(SHAP_FEATURE_NAMES, shap_values)
    }

    # Log SHAP for rollback correlation
    shap_logger.log(
        shap_values=shap_dict,
        rollback_active=model_manager.rollback_active
    )

    return ExplainResponse(
        prediction=float(prediction),
        shap_values=shap_dict,
        note="SHAP values indicate per-feature contribution to the prediction"
    )


# =========================================================
# Health Endpoint
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "primary_detector": PRIMARY_DETECTOR_NAME,
        "rollback_active": model_manager.rollback_active,
        "recovery_counter": model_manager.recovery_counter,
    }


