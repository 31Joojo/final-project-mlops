"""
tests/unit/test_model_loader.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import mlflow.sklearn

from app.model_loader import build_model_uri, load_model_from_registry
from app.settings import get_settings

### ------------------------------ Classes ------------------------------ ###
### Class : FakeModel
class FakeModel:
    """
    Mock model used to simulate mlflow-loaded model behavior.

    :param:
        None

    :return:
        FakeModel: simple stub model for testing
    """
    ### Method : predict()
    def predict(self, X):
        """
        Simulates a model prediction by returning a constant class label.

        :param:
            X Any: input features passed to the model

        :return:
            list[int]: list of predicted class labels
        """
        ### Simulate deterministic prediction
        return [1] * len(X)

    ### Method : predict_proba()
    def predict_proba(self, X):
        """
        Simulates probability predictions for binary classification.

        :param:
            X Any: input features passed to the model

        :return:
            list[list[float]]: probability scores for each class
        """
        ### Simulate probability output
        return [[0.2, 0.8] for _ in range(len(X))]

### ------------------------------- Tests ------------------------------- ###
### Test : test_build_model_uri()
def test_build_model_uri():
    """
    Verifies that the mlflow registry URI is correctly constructed.

    :param:
        None

    :return:
        None: asserts correct URI formatting
    """
    assert build_model_uri("chocolate_sales_logreg", "Staging") == "models:/chocolate_sales_logreg/Staging"

### Test : test_load_model_uses_expected_uri()
def test_load_model_uses_expected_uri(monkeypatch):
    """
    Verifies that load_model_from_registry calls mlflow with the expected URI
    and returns correct metadata.

    :param:
        monkeypatch pytest.MonkeyPatch: fixture used to mock environment variables and mlflow loading

    :return:
        None: asserts correct URI usage and metadata assignment
    """
    ### Override environment variables for model selection
    monkeypatch.setenv("MODEL_NAME", "chocolate_sales_logreg")
    monkeypatch.setenv("MODEL_STAGE", "Staging")

    ### Clear cached settings to reload env variables
    get_settings.cache_clear()
    settings = get_settings()

    captured = {"uri": None}

    ### Mock mlflow sklearn loader
    ### Sub-fonction : fake_load_model()
    def fake_load_model(uri: str):
        """
        Mocks the mlflow model loading function and captures the model URI.

        :param:
            uri str: mlflow model URI used for loading

        :return:
            FakeModel: mocked model instance
        """
        captured["uri"] = uri
        return FakeModel()

    monkeypatch.setattr(mlflow.sklearn, "load_model", fake_load_model)

    ### Load model via registry logic
    loaded = load_model_from_registry(settings=settings, prefer_sklearn=True)

    ### Assert correct registry URI was used
    assert captured["uri"] == "models:/chocolate_sales_logreg/Staging"

    ### Assert metadata correctness
    assert loaded.meta.model_name == "chocolate_sales_logreg"
    assert loaded.meta.model_stage == "Staging"
    assert loaded.meta.flavor == "sklearn"
