"""
tests/integration/test_health.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import pytest
import mlflow.sklearn
from fastapi.testclient import TestClient

from app.settings import get_settings
from app.main import app

### ------------------------------ Classes ------------------------------ ###
### Class : FakeModel
class FakeModel:
    """
    Mock model used to simulate mlflow-loaded model during health endpoint testing.

    :param:
        None

    :return:
        FakeModel: stub model implementing predict and predict_proba
    """
    ### Method : predict()
    def predict(self, X):
        """
        Simulates deterministic class predictions.

        :param:
            X Any: input features

        :return:
            list[int]: predicted labels
        """
        ### Return constant prediction
        return [1] * len(X)

    ### Method : predict_proba()
    def predict_proba(self, X):
        """
        Simulates probability predictions for binary classification.

        :param:
            X Any: input features

        :return:
            list[list[float]]: probability distribution per sample
        """
        ### Return fixed probability distribution
        return [[0.1, 0.9] for _ in range(len(X))]

### ------------------------------- Tests ------------------------------- ###
### Test : test_health_ok()
@pytest.mark.integration
def test_health_ok(monkeypatch):
    """
    Verifies that the health endpoint reports correct model loading metadata
    when the registry load is mocked.

    :param:
        monkeypatch pytest.MonkeyPatch: fixture used to override environment variables and mlflow loading

    :return:
        None: asserts correct health response structure and values
    """
    ### Configure environment variables for model selection
    monkeypatch.setenv("MODEL_NAME", "chocolate_sales_logreg")
    monkeypatch.setenv("MODEL_STAGE", "Staging")

    ### Ensure no mlflow credentials are required
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    ### Clear cached settings to reload environment variables
    get_settings.cache_clear()

    ### Mock mlflow registry loading
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: FakeModel())

    with TestClient(app) as client:
        ### Call health endpoint
        r = client.get("/health")
        assert r.status_code == 200

        data = r.json()

        ### Validate health response structure
        assert data["status"] == "ok"
        assert "loaded" in data
        assert data["model_name"] == "chocolate_sales_logreg"
        assert data["model_stage"] == "Staging"
        assert "model_version" in data
