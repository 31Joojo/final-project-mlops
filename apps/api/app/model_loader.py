"""
apps/api/app/model_loader.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn

from .settings import Settings

logger = logging.getLogger(__name__)

### ------------------------------ Classes ------------------------------ ###
### Class : ModelMeta
@dataclass(frozen=True)
class ModelMeta:
    """
    Represents metadata describing a model loaded from mlflow Model Registry.

    :param:
        model_name str: registered model name
        model_stage str: registry stage used for loading
        model_version Optional[str]: resolved model version
        model_uri str: full mlflow model URI
        flavor str: loading flavor used, sklearn or pyfunc

    :return:
        ModelMeta: immutable metadata container
    """
    model_name: str
    model_stage: str
    model_version: Optional[str]
    model_uri: str
    flavor: str

### Class : LoadedModel
@dataclass
class LoadedModel:
    """
    Encapsulates a loaded mlflow model together with its metadata.

    :param:
        model Any: loaded model object
        meta ModelMeta: associated model metadata

    :return:
        LoadedModel: bundle of model and metadata
    """
    model: Any
    meta: ModelMeta

### ------------------------------ Helpers ------------------------------ ###
### Helper : build_model_uri()
def build_model_uri(model_name: str, model_stage: str) -> str:
    """
    Builds a mlflow registry URI for a given model name and stage.

    :param:
        model_name str: registered model name
        model_stage str: registry stage

    :return:
        str: mlflow model URI in the format models:/<name>/<stage>
    """
    ### Construct mlflow registry URI
    return f"models:/{model_name}/{model_stage}"

### Helper : configure_mlflow()
def configure_mlflow(settings: Settings) -> None:
    """
    Configures mlflow tracking and registry URIs from runtime settings.

    :param:
        settings Settings: application configuration containing mlflow URIs

    :return:
        None
    """
    ### Apply tracking and registry URIs from environment configuration
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_registry_uri(settings.mlflow_tracking_uri)

### Helper : _can_fetch_version()
def _can_fetch_version(settings: Settings) -> bool:
    """
    Determines whether model version fetching from the registry is possible.

    :param:
        settings Settings: application configuration with mlflow credentials

    :return:
        bool: True if version lookup is allowed
    """
    return bool(
        settings.mlflow_tracking_uri
        and settings.mlflow_tracking_username
        and settings.mlflow_tracking_password
    )

### Helper : _try_get_model_version()
def _try_get_model_version(settings: Settings, model_name: str, model_stage: str) -> Optional[str]:
    """
    Attempts to retrieve the model version associated with a given stage.

    :param:
        settings Settings: application configuration with mlflow access
        model_name str: registered model name
        model_stage str: registry stage

    :return:
        Optional[str]: model version string if available
    """
    if not _can_fetch_version(settings):
        return None

    ### Attempt registry lookup via mlflow
    try:
        client = MlflowClient()
        latest = client.get_latest_versions(model_name, stages=[model_stage])
        if latest:
            return str(latest[0].version)

        ### Return latest version for given stage if available
        return None
    except Exception:
        logger.exception("Could not fetch model version from MLflow registry (non-fatal).")
        return None

### Helper : load_model_from_registry()
def load_model_from_registry(settings: Settings, prefer_sklearn: bool = True) -> LoadedModel:
    """
    Loads a model from mlflow model registry based on configured model name and stage.

    :param:
        settings Settings: application configuration with registry parameters
        prefer_sklearn bool: whether to attempt sklearn flavor loading first

    :return:
        LoadedModel: loaded model object bundled with metadata
    """
    configure_mlflow(settings)

    ### Ensure mlflow is configured before loading model
    model_name = settings.model_name
    model_stage = settings.model_stage
    model_uri = build_model_uri(model_name, model_stage)

    model = None
    flavor = "pyfunc"

    ### Try loading via sklearn flavor first
    if prefer_sklearn:
        try:
            model = mlflow.sklearn.load_model(model_uri)
            flavor = "sklearn"
            logger.info("Loaded model via sklearn flavor: %s", model_uri)

        ### Fallback to generic pyfunc flavor if sklearn loading fails
        except Exception:
            logger.info("Could not load sklearn flavor, falling back to pyfunc: %s", model_uri, exc_info=True)

    ### Retrieve model version metadata
    if model is None:
        model = mlflow.pyfunc.load_model(model_uri)
        flavor = "pyfunc"
        logger.info("Loaded model via pyfunc flavor: %s", model_uri)

    version = _try_get_model_version(settings, model_name, model_stage)

    meta = ModelMeta(
        model_name=model_name,
        model_stage=model_stage,
        model_version=version,
        model_uri=model_uri,
        flavor=flavor,
    )
    return LoadedModel(model=model, meta=meta)
