"""
apps/api/app/main.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .predictor import dummy_classifier
from .schema import PredictRequest, PredictResponse
from .settings import Settings, get_settings

### ------------------------------ Helpers ------------------------------ ###
### Helper : _parse_origins()
def _parse_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]

### ----------------------------- Functions ----------------------------- ###
### Function : lifespan()
@asynccontextmanager
async def lifespan(app: FastAPI):
    ### Configure application components at startup
    settings = get_settings()

    ### Add cors middleware using environment-defined origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_origins(settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ### Application runs after middleware setup
    yield

### Initialize application
app = FastAPI(title="MLOps Final Project", lifespan=lifespan)

### Helper : health()
@app.get("/health")
def health() -> dict:
    """
    Provides a healthcheck endpoint to verify that the API is running correctly.

    :return:
        dict: a dictionary containing the service status
    """
    ### Healthcheck endpoint used by Docker and monitoring systems
    return {"status": "ok"}

### Function : predict()
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, settings: Settings = Depends(get_settings)) -> PredictResponse:
    """
    Executes a prediction request using the configured model settings.

    :param:
        payload PredictRequest: validated input features used for inference:
        settings Settings: runtime configuration injected from environment variables
    :return:
        PredictResponse: prediction result including model metadata
    """
    ### Execute inference using the current model configuration
    pred, proba = dummy_classifier(payload)

    ### Returns response
    return PredictResponse(
        prediction=pred,
        probability=round(proba, 6),
        model_stage=settings.model_stage,
        model_version=settings.model_version,
    )
