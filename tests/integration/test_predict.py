"""
tests/integration/test_predict.py

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
    Mock mlflow model used to simulate prediction behavior in integration tests.

    :param:
        None

    :return:
        FakeModel: stub model implementing predict and predict_proba
    """
    classes_ = [0, 1]

    ### Method : predict()
    def predict(self, X):
        """
        Simulates deterministic class predictions.

        :param:
            X Any: input features

        :return:
            list[int]: predicted class labels
        """
        ### Return constant class label
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
### Test : test_predict_ok_returns_expected_shape()
@pytest.mark.integration
def test_predict_ok_returns_expected_shape(monkeypatch):
    """
    Verifies that the predict endpoint returns a correctly structured response
    including model metadata when registry loading is mocked.

    :param:
        monkeypatch pytest.MonkeyPatch: fixture used to override environment variables and mlflow loading

    :return:
        None: asserts correct response structure and values
    """
    ### Configure environment variables for model selection
    monkeypatch.setenv("MODEL_NAME", "chocolate_sales_logreg")
    monkeypatch.setenv("MODEL_STAGE", "Staging")

    ### Ensure mlflow credentials are not required
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    ### Clear cached settings before app initialization
    get_settings.cache_clear()

    ### Mock mlflow registry loading
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: FakeModel())

    with TestClient(app) as client:
        payload = {
            "quantity": 10,
            "unit_price": 15.0,
            "discount": 0.1,
            "country": "FR",
            "product": "Dark Chocolate",
        }

        ### Call predict endpoint
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

        data = r.json()

        ### Validate response structure
        assert set(data.keys()) == {
            "prediction",
            "probability",
            "model_name",
            "model_stage",
            "model_version",
        }

        ### Validate metadata correctness
        assert data["model_name"] == "chocolate_sales_logreg"
        assert data["model_stage"] == "Staging"

        ### Validate prediction
        assert data["prediction"] in (0, 1)
