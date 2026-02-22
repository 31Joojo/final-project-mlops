"""
tests/integration/test_health.py

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
### Test : test_health_ok()
@pytest.mark.integration
def test_health_ok():
    """
    Verifies that the health endpoint is accessible and returns the expected status.

    :returns:
        None: asserts that the API healthcheck responds with status 200 and correct payload
    """
    ### Create a test client for the FastAPI application
    client = TestClient(app)

    ### Send GET request to health endpoint
    r = client.get("/health")

    ### Assert HTTP status code is successful
    assert r.status_code == 200

    ### Assert JSON response matches expected payload
    assert r.json() == {"status": "ok"}
