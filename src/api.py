# uvicorn src.api:app --reload --port 8000
# http://127.0.0.1:8000/docs --> Open In Chrome using Swagger UI


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import numpy as np
import shap
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from src.drift import MovingAverageDriftDetector
from src.drift_adaptive import ADWINDriftDetector, DDMDriftDetector
from src.rollback import ModelManager
from src.drift_logger import DriftLogger
from src.shap_logger import ShapLogger

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
    allow_origins=[
        "https://netra-1-0-trustworthy-secure-nwdaf.vercel.app",
        "https://netra-1-0-trustworthy-sec-nwdaf.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Globals (LAZY LOADED)
# =========================================================
MODEL_VERSION = "FL-v1.0"
FL_MODEL_PATH = Path("models/fl_global_model.pkl")

_model_manager: ModelManager | None = None
_shap_explainer = None

# =========================================================
# Drift Detectors
# =========================================================
ma_detector = MovingAverageDriftDetector(
    window_size=50,
    threshold=0.3,
    min_drift_windows=3,
)

adwin_detector = ADWINDriftDetector(delta=0.002)
ddm_detector = DDMDriftDetector()

PRIMARY_DETECTOR_NAME = "DDM"
drift_logger = DriftLogger("drift_events.csv")
shap_logger = ShapLogger()

# =========================================================
# Utilities
# =========================================================
SLICE_MAP = {"eMBB": 0, "URLLC": 1, "mMTC": 2}
scaler = StandardScaler()  # consistency only

SHAP_FEATURE_NAMES = [
    "time_of_day",
    "slice_type",
    "jitter",
    "packet_loss",
    "throughput",
]

# =========================================================
# Lazy Loaders (CRITICAL FIX)
# =========================================================
def get_model_manager() -> ModelManager:
    global _model_manager

    if _model_manager is None:
        if not FL_MODEL_PATH.exists():
            raise RuntimeError("Model file not found")

        fl_model_data = joblib.load(FL_MODEL_PATH)

        fl_model = SGDRegressor()
        fl_model.coef_ = np.array(fl_model_data["coef"])
        fl_model.intercept_ = np.array(fl_model_data["intercept"])

        _model_manager = ModelManager(
            stable_model=fl_model,
            primary_model=fl_model,
        )

    return _model_manager


def get_shap_explainer(model):
    global _shap_explainer

    if _shap_explainer is None:
        background = np.zeros((50, 6))
        masker = shap.maskers.Independent(background)

        _shap_explainer = shap.LinearExplainer(
            model,
            masker,
        )

    return _shap_explainer


# =========================================================
# Schemas
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
# Helper Functions
# =========================================================
def build_feature_vector(sample: TrafficSample) -> np.ndarray:
    slice_encoded = SLICE_MAP.get(sample.slice_type, 0)

    return np.array([[
        sample.time_of_day,
        slice_encoded,
        sample.jitter,
        sample.packet_loss,
        sample.throughput,
        0.0,  # placeholder (training alignment)
    ]])


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


# =========================================================
# Prediction Endpoint
# =========================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(sample: TrafficSample):
    model_manager = get_model_manager()

    X = build_feature_vector(sample)
    sla_metric = sample.jitter + sample.packet_loss

    ma_drift = ma_detector.update(sla_metric)
    adwin_drift = adwin_detector.update(sla_metric)
    ddm_drift = ddm_detector.update(sla_metric)

    recovered = False

    if ma_drift and not model_manager.rollback_active:
        model_manager.rollback()
        drift_logger.log(
            detector_name=PRIMARY_DETECTOR_NAME,
            sla_metric=sla_metric,
            model_version=MODEL_VERSION,
            action="rollback",
        )

    recovered = model_manager.try_recover(system_stable=not ma_drift)

    if recovered:
        drift_logger.log(
            detector_name=PRIMARY_DETECTOR_NAME,
            sla_metric=sla_metric,
            model_version=MODEL_VERSION,
            action="recovered",
        )

    prediction = model_manager.predict(X)[0]

    warning = None
    if model_manager.rollback_active:
        warning = "⚠ Concept drift detected. Rolled back to stable model."
    elif recovered:
        warning = "✅ System stabilized. Primary model restored."
    elif adwin_drift or ddm_drift:
        warning = "ℹ Early drift warning detected."

    return PredictionResponse(
        predicted_future_load=float(prediction),
        explanation=generate_explanation(sample),
        model_version=f"{MODEL_VERSION} ({model_manager.status()})",
        warning=warning,
    )


# =========================================================
# Explain Endpoint
# =========================================================
@app.post("/explain", response_model=ExplainResponse)
def explain(sample: TrafficSample):
    model_manager = get_model_manager()

    X = build_feature_vector(sample)
    prediction = model_manager.predict(X)[0]

    explainer = get_shap_explainer(model_manager.primary_model)
    shap_values_full = explainer.shap_values(X)
    shap_values = shap_values_full[0][:-1]  # drop placeholder

    shap_dict = {
        name: float(val)
        for name, val in zip(SHAP_FEATURE_NAMES, shap_values)
    }

    shap_logger.log(
        shap_values=shap_dict,
        rollback_active=model_manager.rollback_active,
    )

    return ExplainResponse(
        prediction=float(prediction),
        shap_values=shap_dict,
        note="SHAP values indicate per-feature contribution to the prediction",
    )


# =========================================================
# Health Endpoint (FAST)
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




