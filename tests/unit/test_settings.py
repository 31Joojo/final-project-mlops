"""
tests/unit/test_settings.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import pytest
from app.settings import Settings

### ------------------------------- Tests ------------------------------- ###
### Test : test_settings_from_env()
@pytest.mark.unit
def test_settings_from_env(monkeypatch):
    """
    Verifies that Settings correctly loads configuration values from environment variables.

    :param:
        monkeypatch pytest.MonkeyPatch: fixture used to temporarily set environment variables

    :returns:
        None: asserts environment variables override default configuration values
    """

    ### Set temporary environment variables
    monkeypatch.setenv("MODEL_STAGE", "Staging")
    monkeypatch.setenv("MODEL_NAME", "chocolate_sales_logreg")
    monkeypatch.setenv("ALLOW_MODEL_RELOAD", "1")

    ### Instantiate settings to load environment values
    s = Settings()

    ### Assert values are read from environment instead of defaults
    assert s.model_stage == "Staging"
    assert s.model_name == "chocolate_sales_logreg"
    assert s.allow_model_reload is True
