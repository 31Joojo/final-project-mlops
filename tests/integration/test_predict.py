"""
tests/integration/test_predict.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import pytest
from fastapi.testclient import TestClient

from app.main import app

### ------------------------------- Tests ------------------------------- ###
### Test : test_predict_ok_returns_expected_shape()
@pytest.mark.integration
def test_predict_ok_returns_expected_shape():
    """
    Verifies that the predict endpoint returns a valid response structure for a correct payload.

    :returns:
        None: asserts correct HTTP status and response schema
    """
    ### Initialize test client
    client = TestClient(app)

    ### Valid prediction payload
    payload = {
        "quantity": 10,
        "unit_price": 15.0,
        "discount": 0.1,
        "country": "FR",
        "product": "Dark Chocolate",
    }

    ### Send POST request to predict endpoint
    r = client.post("/predict", json=payload)
    ### Assert request was successful
    assert r.status_code == 200

    data = r.json()

    ### Assert response contains expected keys
    assert set(data.keys()) == {"prediction", "probability", "model_stage", "model_version"}

    ### Assert prediction is binary
    assert data["prediction"] in (0, 1)

    ### Assert probability is within valid range
    assert 0.0 <= data["probability"] <= 1.0

    ### Assert model metadata fields are strings
    assert isinstance(data["model_stage"], str)
    assert isinstance(data["model_version"], str)

### Test : test_predict_validation_error_422()
@pytest.mark.integration
def test_predict_validation_error_422():
    """
    Verifies that the predict endpoint returns a 422 error when input validation fails.

    :returns:
        None: asserts that invalid payload triggers validation error
    """
    ### Initialize test client
    client = TestClient(app)

    ### Invalid payload violating multiple schema constraints
    payload = {
        "quantity": 0,
        "unit_price": -1.0,
        "discount": 2.0,
        "country": "F",
        "product": "",
    }

    ### Send POST request with invalid data
    r = client.post("/predict", json=payload)

    ### Assert validation error status code
    assert r.status_code == 422
