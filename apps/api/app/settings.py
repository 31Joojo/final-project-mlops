"""
apps/api/app/settings.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
### Optional
from functools import lru_cache

### ------------------------------- Class ------------------------------- ###
### Class : Settings
class Settings(BaseSettings):
    """
    Defines runtime configuration loaded from environment variables following the 12-factor app principle.

    :param:
        app_name str: name of the API application
        model_stage str: mlflow model stage to load
        model_version str: specific mlflow model version to load
        cors_allow_origins str: allowed CORS origins for frontend requests

    :returns:
        Settings: an instance containing the resolved configuration values
    """
    app_name: str = "MLOps Final Project API"
    model_stage: str = "dev"
    model_version: str = "0"
    cors_allow_origins: str = "http://localhost:3000"

    ### Avoid warnings with fields starting with "model_"
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        protected_namespaces=(),
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
