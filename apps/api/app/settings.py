"""
apps/api/app/settings.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

from functools import lru_cache
from typing import Optional
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3] if len(_THIS_FILE.parents) > 3 else _THIS_FILE.parents[-1]

_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"
_ENV_FILE = str(_DEFAULT_ENV_FILE) if _DEFAULT_ENV_FILE.exists() else None

### ------------------------------- Class ------------------------------- ###
### Class : Settings
class Settings(BaseSettings):
    """
    Defines runtime configuration loaded from environment variables following the 12-factor app principle.

    :param:
        app_name str: name of the API application
        model_name str: mlflow registered model name
        model_stage str: mlflow model stage to load
        mlflow_tracking_uri Optional[str]: mlflow tracking and registry URI
        mlflow_tracking_username Optional[str]: username for mlflow authentication
        mlflow_tracking_password Optional[str]: password for mlflow authentication
        allow_model_reload bool: enables runtime model reloading
        mlops_disable_model_load bool: disables automatic model loading
        cors_allow_origins str: allowed CORS origins for frontend requests

    :returns:
        Settings: an instance containing the resolved configuration values
    """
    app_name: str = "MLOps Final Project"

    ### Model selector
    model_name: str = "chocolate_sales_logreg"
    model_stage: str = "Staging"

    ### Mlflow and dagshub connection
    mlflow_tracking_uri: Optional[str] = None
    mlflow_tracking_username: Optional[str] = None
    mlflow_tracking_password: Optional[str] = None

    ### Controls
    allow_model_reload: bool = False
    mlops_disable_model_load: bool = False

    ### Web
    cors_allow_origins: str = "http://localhost:3000"

    ### Avoid warnings with fields starting with "model_"
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        protected_namespaces=(),
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

### ------------------------------ Helpers ------------------------------ ###
### Helper : get_settings()
@lru_cache
def get_settings() -> Settings:
    """
    Creates and returns the application runtime settings instance.

    :return:
        Settings: configuration object populated from environment variables
    """
    return Settings()
