"""
tests/integration/test_options.py

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
### Class : FakeOneHot
class FakeOneHot:
    """
    Mock OneHotEncoder providing predefined learned categories.

    :param:
        None

    :return:
        FakeOneHot: stub object exposing categories_
    """
    categories_ = [
        ["Alice", "Bob"],
        ["FR", "USA"],
        ["Dark Chocolate"],
    ]

### Class : FakeCatPipeline
class FakeCatPipeline:
    """
    Mock categorical pipeline containing a fake OneHotEncoder.

    :param:
        None

    :return:
        FakeCatPipeline: stub pipeline with named_steps
    """
    named_steps = {"onehot": FakeOneHot()}

### Class : FakePreprocess
class FakePreprocess:
    """
    Mock ColumnTransformer exposing categorical and numerical transformers.

    :param:
        None

    :return:
        FakePreprocess: stub transformer with transformers_ attribute
    """
    transformers_ = [
        ("num", object(), ["Boxes Shipped", "year", "month", "dayofweek"]),
        ("cat", FakeCatPipeline(), ["Sales Person", "Country", "Product"]),
    ]

### Class: FakeModel
class FakeModel:
    """
    Mock sklearn Pipeline exposing a preprocessing step.

    :param:
        None

    :return:
        FakeModel: stub model compatible with Predictor
    """
    named_steps = {"preprocess": FakePreprocess()}

    ### Method : predict()
    def predict(self, X):
        """
        Simulates deterministic class predictions.

        :param:
            X Any: input feature matrix

        :return:
            list[int]: predicted class labels
        """

    ### Method : predict_proba()
    def predict_proba(self, X):
        """
        Simulates probability predictions for binary classification.

        :param:
            X Any: input feature matrix

        :return:
            list[list[float]]: probability scores for each class
        """
        return [[0.2, 0.8] for _ in range(len(X))]

### ------------------------------- Tests ------------------------------- ###
### Test : test_options_returns_lists()
@pytest.mark.integration
def test_options_returns_lists(monkeypatch):
    """
    Verifies that the /options endpoint returns categorical values
    extracted from the loaded preprocessing pipeline.

    :param:
        monkeypatch pytest.MonkeyPatch: fixture used to override environment variables and mlflow loading

    :return:
        None: asserts correct JSON response
    """
    ### Configure model selection environment variables
    monkeypatch.setenv("MODEL_NAME", "chocolate_sales_logreg")
    monkeypatch.setenv("MODEL_STAGE", "@staging")

    ### Ensure no MLflow credentials are required
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    ### Clear cached settings to reload environment variables
    get_settings.cache_clear()

    ### Mock mlflow model loading
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: FakeModel())

    with TestClient(app) as client:
        ### Call /options endpoint
        r = client.get("/options")
        assert r.status_code == 200

        data = r.json()

        ### Validate response structure
        assert set(data.keys()) == {"sales_persons", "countries", "products"}

        ### Validate returned categorical values
        assert "Alice" in data["sales_persons"]
        assert "FR" in data["countries"]
        assert "Dark Chocolate" in data["products"]

### Test : test_options_503_when_model_not_loaded()
@pytest.mark.integration
def test_options_503_when_model_not_loaded(monkeypatch):
    """
    Verifies that the /options endpoint returns 503
    when the model is not loaded at startup.

    :param:
        monkeypatch pytest.MonkeyPatch: fixture used to disable model loading

    :return:
        None: asserts 503 status code
    """
    ### Disable model loading at startup
    monkeypatch.setenv("MLOPS_DISABLE_MODEL_LOAD", "1")

    ### Clear cached settings to apply environment change
    get_settings.cache_clear()

    with TestClient(app) as client:
        ### Call /options endpoint without loaded model
        r = client.get("/options")

        ### Expect service unavailable status
        assert r.status_code == 503
