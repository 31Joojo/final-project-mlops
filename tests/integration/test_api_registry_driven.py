"""
tests/integration/test_api_registry_driven.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import mlflow.sklearn
from fastapi.testclient import TestClient

from app.main import app
from app.settings import get_settings

### ------------------------------ Classes ------------------------------ ###
### Class : FakeModel
class FakeModel:
    """
    Mock mlflow model used to simulate registry-loaded model behavior.

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
            X Any: input features passed to the model

        :return:
            list[int]: predicted class labels
        """
        return [1] * len(X)

    ### Method : predict_proba
    def predict_proba(self, X):
        """
        Simulates probability predictions for binary classification.

        :param:
            X Any: input features passed to the model

        :return:
            list[list[float]]: class probability scores
        """
        return [[0.1, 0.9] for _ in range(len(X))]

### ------------------------------- Tests ------------------------------- ###
### Test : test_health_and_predict_mocked_registry_load()
def test_health_and_predict_mocked_registry_load(monkeypatch):
    """
    Verifies that the API loads a model from the registry
    and correctly serves health and prediction endpoints.

    :param:
        monkeypatch pytest.MonkeyPatch: fixture used to override environment variables and mlflow loading

    :return:
        None: asserts successful health check and prediction behavior
    """
    ### Set environment variables for model selection
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
        ### Verify health endpoint
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()

        assert body["status"] == "ok"
        assert body["loaded"] is True
        assert body["model_name"] == "chocolate_sales_logreg"
        assert body["model_stage"] == "Staging"

        ### Verify prediction endpoint
        payload = {
            "quantity": 10,
            "unit_price": 2.5,
            "discount": 0.1,
            "country": "FR",
            "product": "Dark",
        }

        r2 = client.post("/predict", json=payload)
        assert r2.status_code == 200

        out = r2.json()
        assert out["prediction"] == 1
        assert out["probability"] == 0.9
        assert out["model_name"] == "chocolate_sales_logreg"
        assert out["model_stage"] == "Staging"
