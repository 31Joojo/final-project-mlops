"""
tests/e2e/test_staging_live.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import os
import datetime as dt
import pytest
import requests


pytestmark = pytest.mark.e2e

### ------------------------------ Helpers ------------------------------ ###
### Helper : _base_url()
def _base_url() -> str:
    """
    Retrieves the staging API base URL from environment variables.

    :param:
        None

    :return:
        str: base URL of the staging API without trailing slash
    """
    ### Read base URL from environment
    base = os.environ["STAGING_API_BASE_URL"].rstrip("/")

    return base

### Helper : _pick()
def _pick(options: dict, keys: list[str], fallback):
    """
    Selects the first available value from multiple possible keys in an options dictionary.

    :param:
        options dict: dictionary returned by /options endpoint
        keys list[str]: possible key variations to search for
        fallback Any: default value if no valid option is found

    :return:
        Any: selected option value or fallback
    """
    ### Try multiple key variants for robustness
    for k in keys:
        v = options.get(k)
        if isinstance(v, list) and len(v) > 0:
            return v[0]

    return fallback

### ------------------------------- Tests ------------------------------- ###
### Test : test_staging_health_options_predict()
def test_staging_health_options_predict():
    """
    End-to-end live test against the staging API.

    This test verifies:
        - /health endpoint availability
        - /options endpoint availability
        - /predict endpoint functionality
        - minimal response contract integrity

    :param:
        None

    :return:
        None: asserts correct staging API behavior
    """
    base = _base_url()
    timeout = float(os.getenv("E2E_HTTP_TIMEOUT", "10"))

    ### Test /health endpoint
    r = requests.get(f"{base}/health", timeout=timeout)
    assert r.status_code == 200, r.text
    health = r.json()
    assert isinstance(health, dict)

    ### Test /options endpoint
    r = requests.get(f"{base}/options", timeout=timeout)
    assert r.status_code == 200, r.text
    options = r.json()
    assert isinstance(options, dict)

    ### Extract categorical values dynamically
    sales_person = _pick(options, ["Sales Person", "sales_person", "salesPerson"], "Alice")
    country = _pick(options, ["Country", "country"], "FR")
    product = _pick(options, ["Product", "product"], "Dark Chocolate")

    ### Stable date features
    d = dt.date(2026, 2, 27)
    year = d.year
    month = d.month
    dayofweek = d.weekday()

    ### Try multiple payload formats for compatibility
    payload_space_keys = {
        "Sales Person": sales_person,
        "Country": country,
        "Product": product,
        "Boxes Shipped": 10,
        "year": year,
        "month": month,
        "dayofweek": dayofweek,
    }
    payload_snake_case = {
        "sales_person": sales_person,
        "country": country,
        "product": product,
        "boxes_shipped": 10,
        "year": year,
        "month": month,
        "dayofweek": dayofweek,
    }

    ### Test /predict endpoint
    r = requests.post(f"{base}/predict", json=payload_space_keys, timeout=timeout)

    ### Fallback to snake_case format if validation fails
    if r.status_code == 422:
        r = requests.post(f"{base}/predict", json=payload_snake_case, timeout=timeout)

    assert r.status_code == 200, r.text
    out = r.json()
    assert isinstance(out, dict)

    ### Minimal contract validation
    assert "prediction" in out
    assert "probability" in out
    assert 0.0 <= float(out["probability"]) <= 1.0
    assert "model_name" in out
    assert "model_stage" in out
