"""
apps/api/app/main.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import load_model_from_registry
from .predictor import ModelNotLoadedError, Predictor
from .schema import PredictRequest, PredictResponse
from .settings import Settings, get_settings

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

### ------------------------------ Helpers ------------------------------ ###
### Helper : _parse_origins()
def _parse_origins(raw: str) -> list[str]:
    """
    Parses a comma-separated string of origins into a list.

    :param:
        raw str: comma-separated origins string

    :return:
        list[str]: cleaned list of origins
    """
    ### Split and clean CORS origins string
    return [o.strip() for o in raw.split(",") if o.strip()]

### Helper : _bool()
def _bool(v: Any) -> bool:
    """
    Converts various truthy string representations into a boolean value.

    :param:
        v Any: value to interpret as boolean

    :return:
        bool: True if value represents a truthy value
    """
    if v is None:
        return False

    ### Interpret common truthy values
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

### ----------------------------- Functions ----------------------------- ###
### Function : lifespan()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes application state at startup and loads the model from mlflow registry.

    :param:
        app FastAPI: FastAPI application instance

    :return:
        None
    """
    settings = get_settings()

    predictor = Predictor()

    ### Initialize in-memory predictor instance
    app.state.predictor = predictor

    ### Store predictor in application state
    if _bool(settings.mlops_disable_model_load):
        logger.warning("MLOPS_DISABLE_MODEL_LOAD enabled -> startup continues with loaded=False")
        yield
        return

    try:
        loaded = load_model_from_registry(settings=settings, prefer_sklearn=True)
        predictor.set_model(loaded)
        logger.info(
            "Model loaded: name=%s stage=%s version=%s uri=%s flavor=%s",
            loaded.meta.model_name,
            loaded.meta.model_stage,
            loaded.meta.model_version,
            loaded.meta.model_uri,
            loaded.meta.flavor,
        )
    except Exception:
        logger.exception("Model failed to load at startup. API starts with loaded=False.")

    ### Continue startup even if model loading fails
    yield

### Function : create_app()
def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application including middleware,
    model loading lifecycle, and API routes.

    :param:
        None

    :return:
        FastAPI: fully configured FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(title="MLOps Final Project", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_origins(settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health(settings: Settings = Depends(get_settings)) -> dict:
        """
        Provides application health status and model metadata.

        :param:
            settings Settings: runtime configuration

        :return:
            dict: health status including model loading information
        """
        predictor: Predictor = app.state.predictor

        ### Retrieve predictor metadata
        meta = predictor.meta

        return {
            "status": "ok",
            "loaded": predictor.loaded,
            "model_name": meta.model_name if meta else settings.model_name,
            "model_stage": meta.model_stage if meta else settings.model_stage,
            "model_version": meta.model_version if meta else None,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, settings: Settings = Depends(get_settings)) -> PredictResponse:
        """
        Executes prediction using the in-memory registry-loaded model.

        :param:
            payload PredictRequest: validated input features
            settings Settings: runtime configuration

        :return:
            PredictResponse: prediction result including model metadata
        """
        predictor: Predictor = app.state.predictor

        ### Ensure model is loaded before inference
        if not predictor.loaded:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "model_not_loaded",
                    "hint": "Provide MLflow/DagsHub env vars or mock loader in CI",
                    "model_name": settings.model_name,
                    "model_stage": settings.model_stage,
                    "loaded": False,
                },
            )

        ### Perform prediction using in-memory model
        try:
            out = predictor.predict(payload)
            meta = predictor.meta
            return PredictResponse(
                prediction=out.prediction,
                probability=round(out.probability, 6) if out.probability is not None else None,
                model_name=meta.model_name if meta else settings.model_name,
                model_stage=meta.model_stage if meta else settings.model_stage,
                model_version=meta.model_version if meta else None,
            )
        except ModelNotLoadedError:
            raise HTTPException(status_code=503, detail="Model not loaded")
        except Exception as e:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/options")
    def options():
        """
        Returns available categorical feature values extracted from the trained model pipeline.

        The endpoint inspects the loaded preprocessing pipeline
        to dynamically retrieve known categories for dropdown population.

        :param:
            None

        :return:
            dict: available sales_persons, countries, and products
        """
        predictor: Predictor = app.state.predictor

        ### Ensure model is loaded before accessing pipeline
        if not predictor.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        ### Access underlying sklearn model
        model = predictor._loaded.model
        preprocess = None

        ### Extract preprocessing step from sklearn Pipeline
        if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
            preprocess = model.named_steps["preprocess"]

        ### If no preprocessing pipeline found return empty options
        if preprocess is None:
            return {"sales_persons": [], "countries": [], "products": []}

        ### Locate categorical transformer inside ColumnTransformer
        cat_pipe = None
        for name, transformer, cols in preprocess.transformers_:
            if name == "cat":
                cat_pipe = transformer
                cat_cols = cols
                break

        ### If categorical pipeline missing return empty
        if cat_pipe is None:
            return {"sales_persons": [], "countries": [], "products": []}

        ### Retrieve OneHotEncoder and its learned categories
        ohe = cat_pipe.named_steps.get("onehot")
        if ohe is None or not hasattr(ohe, "categories_"):
            return {"sales_persons": [], "countries": [], "products": []}

        ### Map categorical columns to learned categories
        cats = ohe.categories_
        mapping = dict(zip(cat_cols, cats))

        ### Return sorted values for frontend dropdowns
        return {
            "sales_persons": sorted([str(x) for x in mapping.get("Sales Person", [])]),
            "countries": sorted([str(x) for x in mapping.get("Country", [])]),
            "products": sorted([str(x) for x in mapping.get("Product", [])]),
        }

    @app.post("/reload-model")
    def reload_model(settings: Settings = Depends(get_settings)) -> dict:
        """
        Reloads the model from the mlflow registry when allowed.

        :param:
            settings Settings: runtime configuration

        :return:
            dict: reload status and updated model metadata
        """
        ### Prevent reload in Production or when disabled
        if settings.model_stage == "Production" or not _bool(settings.allow_model_reload):
            raise HTTPException(status_code=403, detail="reload-model disabled in Production")

        try:
            loaded = load_model_from_registry(settings=settings, prefer_sklearn=True)
            predictor: Predictor = app.state.predictor

            ### Replace currently loaded model
            predictor.set_model(loaded)
            return {
                "reloaded": True,
                "model_name": loaded.meta.model_name,
                "model_stage": loaded.meta.model_stage,
                "model_version": loaded.meta.model_version,
            }
        except Exception:
            logger.exception("Reload failed")
            raise HTTPException(status_code=500, detail="Reload failed")

    return app


app = create_app()
