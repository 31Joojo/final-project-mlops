"""
tests/unit/test_predictor.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import numpy as np
import pytest

from app.predictor import Predictor, ModelNotLoadedError
from app.schema import PredictRequest

### ------------------------------ Classes ------------------------------ ###
### Class : FakeModel
class FakeModel:
    """
    Mock model used to simulate sklearn model behavior for Predictor tests.

    :param:
        None

    :return:
        FakeModel: stub model implementing predict and predict_proba
    """
    classes_ = np.array([0, 1])

    ### Method : predict()
    def predict(self, X):
        """
        Simulates deterministic class predictions.

        :param:
            X Any: input feature matrix

        :return:
            np.ndarray: predicted class labels
        """
        ### Return constant class label
        return np.array([1] * len(X))

    ### Method : predict_proba()
    def predict_proba(self, X):
        """
        Simulates probability predictions for binary classification.

        :param:
            X Any: input feature matrix

        :return:
            np.ndarray: probability matrix
        """
        ### Return fixed probability distribution
        return np.array([[0.1, 0.9] for _ in range(len(X))])

### Class : FakeLoadedModel
class FakeLoadedModel:
    """
    Mock LoadedModel object mimicking MLflow registry output.

    :param:
        None

    :return:
        FakeLoadedModel: bundle containing fake model and metadata
    """
    def __init__(self):
        self.model = FakeModel()

        ### Provide fake metadata
        ### Class : _Meta
        class _Meta:
            model_name = "chocolate_sales_logreg"
            model_stage = "Staging"
            model_version = "1"
            model_uri = "models:/chocolate_sales_logreg/Staging"
            flavor = "sklearn"

        self.meta = _Meta()

### ------------------------------- Tests ------------------------------- ###
### Test : test_predictor_raises_if_model_not_loaded()
def test_predictor_raises_if_model_not_loaded():
    """
    Verifies that Predictor raises ModelNotLoadedError
    when predict is called without a loaded model.

    :param:
        None

    :return:
        None: asserts proper exception handling
    """
    ### Initialize predictor without model
    p = Predictor()

    req = PredictRequest(quantity=10, unit_price=2.5, discount=0.1, country="FR", product="Dark")

    ### Expect error when model is not loaded
    with pytest.raises(ModelNotLoadedError):
        p.predict(req)

### Test : test_predictor_predict_and_proba()
def test_predictor_predict_and_proba():
    """
    Verifies that Predictor returns correct prediction and probability
    when a model is loaded.

    :param:
        None

    :return:
        None: asserts correct inference output
    """
    ### Initialize predictor and assign fake model
    p = Predictor()
    p.set_model(FakeLoadedModel())

    req = PredictRequest(
        quantity=10,
        unit_price=2.5,
        discount=0.1,
        country="FR",
        product="Dark",
    )

    ### Make prediction
    out = p.predict(req)

    ### Assert expected inference output
    assert out.prediction == 1
    assert out.probability == 0.9
