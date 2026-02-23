"""
tests/unit/test_predictor.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import pytest

from app.predictor import dummy_classifier, sigmoid
from app.schema import PredictRequest

### ------------------------------- Tests ------------------------------- ###
### Test : test_sigmoid_is_bounded()
@pytest.mark.unit
def test_sigmoid_is_bounded():
    """
    Verifies that the sigmoid function outputs values within the expected probability range.

    :returns:
        None: asserts sigmoid remains bounded between 0 and 1
    """
    ### Assert sigmoid approaches 0 for large negative input
    assert 0.0 <= sigmoid(-50) < 0.01

    ### Assert sigmoid approaches 1 for large positive input
    assert 0.99 < sigmoid(50) <= 1.0

### Test : test_dummy_classifier_changes_with_revenue()
@pytest.mark.unit
def test_dummy_classifier_changes_with_revenue():
    """
    Verifies that the dummy classifier produces higher probabilities for higher revenue inputs.

    :returns:
        None: asserts monotonic behavior of prediction with increasing revenue
    """
    ### Create low-revenue input
    low = PredictRequest(quantity=1, unit_price=1.0, discount=0.0, country="FR", product="X")

    ### Create high-revenue input
    high = PredictRequest(quantity=500, unit_price=50.0, discount=0.0, country="FR", product="X")

    ### Run classifier on both inputs
    pred_low, proba_low = dummy_classifier(low)
    pred_high, proba_high = dummy_classifier(high)

    ### Assert predictions are binary
    assert pred_low in (0, 1)
    assert pred_high in (0, 1)

    ### Assert probability increases with revenue
    assert proba_high > proba_low

    ### Assert classification follows probability trend
    assert pred_high >= pred_low
