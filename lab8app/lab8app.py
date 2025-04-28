# lab8app/lab8app.py
# ---------------------------------------------------------------
# FastAPI wrapper around the RandomForestRegressor stored in MLflow
# ---------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import mlflow

# ---------- CONFIG ------------------------------------------------
MLFLOW_TRACKING_URI = "https://mlflow-server-931658252548.us-west2.run.app"
MODEL_ARTIFACT_URI  = (
    "mlflow-artifacts:/3/666241ac5fa0418e99dc133edb07501d/artifacts/model"
)  # <-- paste **your** run-artifact URI here
N_FEATURES = 30  # model was trained on 30 engineered columns
# ------------------------------------------------------------------

app = FastAPI(
    title="Ames-Housing Price Estimator",
    description="Simple FastAPI service that loads a RandomForestRegressor from MLflow and predicts SalePrice.",
    version="0.1.0",
)

class FeaturesRequest(BaseModel):
    """Expect exactly 30 numeric values, in the same order used at training time."""
    features: list[float] = Field(
        ..., example=[7,1800,1000,960,500,800,9450,2004,484,2,2,
                      2005,511836,65,84,0,6,2004,5,196,0,1380,0,
                      2,0,0,7,1,1,120]
    )

@app.on_event("startup")
def load_model() -> None:
    """Executed once when Uvicorn starts: set tracking URI, download and cache the model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)          # 👈 make mlflow-artifacts work
    global model
    model = mlflow.sklearn.load_model(MODEL_ARTIFACT_URI)
    print(f"✅  Model loaded from {MODEL_ARTIFACT_URI}")

@app.post("/predict")
def predict(req: FeaturesRequest):
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"`features` must contain {N_FEATURES} values, got {len(req.features)}",
        )
    X = np.array([req.features])
    preds = model.predict(X).tolist()
    return {"prediction": preds[0]}
