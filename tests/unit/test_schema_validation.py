"""
tests/unit/test_schema_validation.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import pytest
from pydantic import ValidationError

from app.schema import PredictRequest

### ------------------------------- Tests ------------------------------- ###
### Test : test_predict_request_accepts_valid_payload()
@pytest.mark.unit
def test_predict_request_accepts_valid_payload():
    """
    Verifies that PredictRequest accepts a valid payload and stores correct values.

    :returns:
        None: asserts valid input is correctly instantiated
    """
    ### Create valid request instance
    req = PredictRequest(
        quantity=10,
        unit_price=12.5,
        discount=0.1,
        country="FR",
        product="Dark Chocolate",
    )

    ### Assert fields are correctly assigned
    assert req.quantity == 10
    assert req.unit_price == 12.5
    assert req.discount == 0.1
    assert req.country == "FR"
    assert req.product == "Dark Chocolate"

### Test : test_predict_request_rejects_out_of_bounds_values()
@pytest.mark.unit
def test_predict_request_rejects_out_of_bounds_values():
    """
    Verifies that PredictRequest raises a ValidationError for invalid field values.

    :returns:
        None: asserts invalid input triggers schema validation error
    """
    ### Invalid values should raise validation error
    with pytest.raises(ValidationError):
        PredictRequest(
            quantity=0,
            unit_price=-1.0,
            discount=2.0,
            country="F",
            product="",
        )

### Test : test_predict_request_requires_fields()
@pytest.mark.unit
def test_predict_request_requires_fields():
    """
    Verifies that PredictRequest raises a ValidationError when required fields are missing.

    :returns:
        None: asserts missing required fields trigger validation error
    """
    ### Missing required fields should raise validation error
    with pytest.raises(ValidationError):
        PredictRequest(
            quantity=10,
            unit_price=12.5,
            discount=0.1,
            country="FR",
            ### product missing
        )
