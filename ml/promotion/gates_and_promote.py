"""
ml/promotion/gates_and_promote.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import mlflow
from mlflow.tracking import MlflowClient

### ------------------------------ Classes ------------------------------ ###
### Class : GateResult()
@dataclass
class GateResult:
    passed: bool
    details: Dict[str, Any]

### ------------------------------ Helpers ------------------------------ ###
### Helper : _norm_stage()
def _norm_stage(stage: str) -> str:
    """
    Normalizes a model stage or alias string for safe comparison and display.

    :param:
        stage str: raw stage or alias string

    :return:
        str: normalized stage string
    """
    ### Return as-is if stage is empty or None-like
    if not stage:
        return stage

    ### Remove leading/trailing whitespace
    return stage.strip()

### Helper : _get_env_float()
def _get_env_float(name: str, default: float) -> float:
    """
    Retrieves a float value from environment variables with a fallback default.

    :param:
        name str: environment variable name
        default float: default value if variable is missing or empty

    :return:
        float: parsed float value from environment or default
    """
    ### Read environment variable
    val = os.getenv(name)

    ### Return parsed float if defined and non-empty, otherwise fallback to default
    return float(val) if val is not None and val != "" else default

### Helper : _get_env_int()
def _get_env_int(name: str, default: int) -> int:
    """
    Retrieves an integer value from environment variables with a fallback default.

    :param:
        name str: environment variable name
        default int: default value if variable is missing or empty

    :return:
        int: parsed integer value from environment or default
    """
    ### Read environment variable
    val = os.getenv(name)

    ### Return parsed integer if defined and non-empty
    return int(val) if val is not None and val != "" else default

### Helper : _p95_ms()
def _p95_ms(values_s: list[float]) -> float:
    """
    Computes the 95th percentile latency in milliseconds.

    :param:
        values_s list[float]: list of latency values in seconds

    :return:
        float: 95th percentile latency in milliseconds
    """
    ### Convert seconds to milliseconds
    arr = np.array(values_s, dtype=float) * 1000.0

    ### Compute 95th percentile
    return float(np.percentile(arr, 95))

### Helper : _mlflow_client()
def _mlflow_client() -> MlflowClient:
    """
    Instantiates and returns an mlflow client.

    :param:
        None

    :return:
        MlflowClient: mlflow tracking and registry client instance
    """
    ### Create mlflow client using current tracking registry configuration
    return MlflowClient()

### Helper : _get_model_version_by_alias()
def _get_model_version_by_alias(
    client: MlflowClient,
    model_name: str,
    alias: str
) -> Optional[str]:
    """
    Retrieves a model version from the mlflow registry using an alias.

    :param:
        client MlflowClient: initialized mlflow client
        model_name str: registered model name
        alias str: alias name

    :return:
        Optional[str]: resolved model version as string
    """
    ### Check if mlflow version supports alias retrieval
    if not hasattr(client, "get_model_version_by_alias"):
        return None

    try:
        ### Attempt to resolve model version from alias
        mv = client.get_model_version_by_alias(
            name=model_name,
            alias=alias
        )
        return str(mv.version)
    except Exception:
        ### Return None if alias lookup fails
        return None

### Helper : _get_prod_baseline()
def _get_prod_baseline(
    client: MlflowClient,
    model_name: str
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Retrieves the current Production baseline model information.

    :param:
        client MlflowClient: initialized mlflow client
        model_name str: registered model name

    :return:
        Tuple[Optional[str], Optional[str], Optional[float]]:
            - production model version
            - associated run_id
            - f1 metric value when available
            Returns (None, None, None) if no production model exists.
    """
    #### Try classic mlflow production stage
    try:
        latest = client.get_latest_versions(model_name, stages=["Production"])

        if latest:
            mv = latest[0]

            ### Extract run_id linked to this version
            run_id = getattr(mv, "run_id", None)

            ### Try retrieving f1 from run metrics
            prod_f1 = _get_run_metric(client, run_id, "f1") if run_id else None

            ### Fallback: retrieve f1 from version tags
            if prod_f1 is None:
                prod_f1 = _get_version_tag_metric(
                    client,
                    model_name,
                    str(mv.version),
                    "f1",
                )

            return str(mv.version), run_id, prod_f1

    except Exception:
        ### Ignore errors and try alias fallback
        pass

    ### Try alias-based production resolution
    v = _get_model_version_by_alias(client, model_name, "production")

    if v:
        mv = client.get_model_version(model_name, v)

        ### Extract run_id
        run_id = getattr(mv, "run_id", None)

        ### Try retrieving f1 from run metrics
        prod_f1 = _get_run_metric(client, run_id, "f1") if run_id else None

        ### Fallback: retrieve f1 from version tags
        if prod_f1 is None:
            prod_f1 = _get_version_tag_metric(
                client,
                model_name,
                v,
                "f1",
            )

        return v, run_id, prod_f1

    ### No production baseline found
    return None, None, None

### Helper : _get_run_metric()
def _get_run_metric(
    client: MlflowClient,
    run_id: Optional[str],
    key: str
) -> Optional[float]:
    """
    Retrieves a metric value from an mlflow run.

    :param:
        client MlflowClient: initialized mlflow client
        run_id Optional[str]: mlflow run identifier
        key str: metric name to retrieve

    :return:
        Optional[float]: metric value if found
    """
    ### Return None if no run_id is provided
    if not run_id:
        return None

    try:
        ### Fetch run information from mlflow
        run = client.get_run(run_id)

        ### Retrieve metric value
        v = run.data.metrics.get(key)

        ### Convert to float if available
        return float(v) if v is not None else None

    except Exception:
        ### Return None if run retrieval fails
        return None

### Helper : _get_version_tag_metric()
def _get_version_tag_metric(
    client: MlflowClient,
    model_name: str,
    version: str,
    key: str
) -> Optional[float]:
    """
    Retrieves a metric value stored as a model version tag in the mlflow registry.

    :param:
        client MlflowClient: initialized mlflow client
        model_name str: registered model name
        version str: model version number
        key str: metric name

    :return:
        Optional[float]: metric value if found
    """
    try:
        ### Retrieve model version metadata from registry
        mv = client.get_model_version(model_name, version)

        ### Extract tags dictionary
        tags = getattr(mv, "tags", {}) or {}

        ### Retrieve metric stored as tag
        v = tags.get(f"metric.{key}")

        ### Convert to float if available
        return float(v) if v is not None else None

    except Exception:
        ### Return None if version lookup or parsing fails
        return None

### Helper : _tag_version()
def _tag_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    tags: Dict[str, Any]
) -> None:
    """
    Attaches multiple tags to a specific mlflow model version.

    :param:
        client MlflowClient: initialized mlflow client
        model_name str: registered model name
        version str: model version number
        tags Dict[str, Any]: dictionary of tag key-value pairs

    :return:
        None
    """
    ### Iterate over provided tags
    for k, v in tags.items():
        try:
            ### Set tag on model version
            client.set_model_version_tag(model_name, version, k, str(v))
        except Exception:
            ### Ignore tagging errors to avoid breaking promotion flow
            pass

### Helper : _api_get_json()
def _api_get_json(url: str, timeout_s: int = 10) -> Any:
    """
    Performs an HTTP GET request and returns the JSON response.

    :param:
        url str: target endpoint URL
        timeout_s int: request timeout in seconds

    :return:
        Any: parsed JSON response body

    :raises:
        requests.HTTPError: if the HTTP request fails
    """
    ### Execute GET request with timeout
    r = requests.get(url, timeout=timeout_s)

    ### Raise exception if response status is not 2xx
    r.raise_for_status()

    ### Return parsed JSON payload
    return r.json()

### Helper : _api_post_json()
def _api_post_json(
    url: str,
    payload: Dict[str, Any],
    timeout_s: int = 20
) -> Any:
    """
    Performs an HTTP POST request with a JSON payload and returns the JSON response.

    :param:
        url str: target endpoint URL
        payload Dict[str, Any]: JSON body to send in the request
        timeout_s int: request timeout in seconds (default=20)

    :return:
        Any: parsed JSON response body

    :raises:
        requests.HTTPError: if the HTTP request fails
    """
    ### Execute POST request with json body and timeout
    r = requests.post(url, json=payload, timeout=timeout_s)

    ### Raise exception if response status is not 2xx
    r.raise_for_status()

    ### Return parsed json payload
    return r.json()

### Helper : _extract_predict_schema_from_openapi()
def _extract_predict_schema_from_openapi(
    openapi: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extracts the JSON request body schema for the POST /predict endpoint
    from a FastAPI OpenAPI specification.

    :param:
        openapi Dict[str, Any]: OpenAPI specification dictionary

    :return:
        Optional[Dict[str, Any]]: request body schema if found
    """
    ### Access OpenAPI paths definition
    paths = openapi.get("paths", {})

    ### Retrieve /predict endpoint definition
    predict = paths.get("/predict", {})

    ### Retrieve POST operation
    post = predict.get("post", {})

    ### Extract request body section
    request_body = post.get("requestBody", {})

    ### Access JSON content definition
    content = request_body.get("content", {})
    app_json = content.get("application/json", {})

    ### Return extracted schema
    schema = app_json.get("schema")
    return schema

### Helper : _resolve_schema_ref()
def _resolve_schema_ref(
    openapi: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Resolves a JSON Schema $ref reference using the OpenAPI components section.

    :param:
        openapi Dict[str, Any]: OpenAPI specification dictionary
        schema Dict[str, Any]: schema object that may contain a $ref

    :return:
        Dict[str, Any]: resolved schema definition
    """
    ### If schema is already concrete -> no reference
    if "$ref" not in schema:
        return schema

    ### Extract referenced schema name
    ref = schema["$ref"]
    name = ref.split("/")[-1]

    ### Resolve reference from OpenAPI components
    return (
        openapi.get("components", {})
        .get("schemas", {})
        .get(name, {})
    )

### Helper : _build_payload_from_schema()
def _build_payload_from_schema(
    schema: Dict[str, Any],
    options_json: Any
) -> Dict[str, Any]:
    """
    Builds a minimal valid JSON payload based on an OpenAPI schema.

    :param:
        schema Dict[str, Any]: OpenAPI schema definition for request body
        options_json Any: json response from /options endpoint

    :return:
        Dict[str, Any]: generated payload compatible with the schema
    """
    ### Extract schema properties and required fields
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    payload: Dict[str, Any] = {}

    ### Normalize /options keys for case-insensitive lookup
    options_map: Dict[str, list] = {}
    if isinstance(options_json, dict):
        for k, v in options_json.items():
            if isinstance(v, list):
                options_map[k.lower()] = v

    ### Helper to select a value from /options when available
    def pick_option(field: str) -> Optional[Any]:
        ### Exact lowercase match
        f = field.lower()
        if f in options_map and options_map[f]:
            return options_map[f][0]

        ### Flexible match
        f2 = f.replace("_", "").replace(" ", "")
        for ok, ov in options_map.items():
            ok2 = ok.replace("_", "").replace(" ", "")
            if ok2 == f2 and ov:
                return ov[0]

        return None

    ### Build payload using required fields first
    for field, spec in props.items():
        if required and field not in required:
            ### Keep payload minimal: include only required fields
            continue

        ftype = (spec.get("type") or "").lower()
        fmt = (spec.get("format") or "").lower()

        ### Prefer /options value when available
        opt = pick_option(field)
        if opt is not None:
            payload[field] = opt
            continue

        ### Infer default value based on schema type/format
        if fmt in {"date", "date-time"} or "date" in field.lower():
            payload[field] = "2021-01-01"
        elif ftype == "integer":
            payload[field] = 1
        elif ftype == "number":
            payload[field] = 1.0
        else:
            payload[field] = "dummy"

    ### Fallback: if no required fields defined, include all properties
    if not payload and props:
        for field, spec in props.items():
            ftype = (spec.get("type") or "").lower()
            fmt = (spec.get("format") or "").lower()

            opt = pick_option(field)
            if opt is not None:
                payload[field] = opt
            elif fmt in {"date", "date-time"} or "date" in field.lower():
                payload[field] = "2021-01-01"
            elif ftype == "integer":
                payload[field] = 1
            elif ftype == "number":
                payload[field] = 1.0
            else:
                payload[field] = "dummy"

    return payload

### Helper : gate_metric_threshold()
def gate_metric_threshold(
    client: MlflowClient,
    model_name: str,
    model_version: str,
    candidate_run_id: Optional[str],
) -> GateResult:
    """
    Validates the metric threshold gate based on F1 score.

    The candidate model passes if:
        - candidate_f1 >= production_f1
        - otherwise candidate_f1 >= GATE_MIN_F1 environment value

    :param:
        client MlflowClient: initialized mlflow client
        model_name str: registered model name
        model_version str: candidate model version
        candidate_run_id Optional[str]: mlflow run id associated with candidate

    :return:
        GateResult: object containing pass/fail status and evaluation details
    """
    ### Retrieve minimum fallback f1 threshold from environment
    min_f1 = _get_env_float("GATE_MIN_F1", 0.50)

    ### Retrieve candidate F1 from run metrics
    cand_f1 = (
        _get_run_metric(client, candidate_run_id, "f1")
        if candidate_run_id
        else None
    )

    ### Fallback: retrieve candidate f1 from version tags
    if cand_f1 is None:
        cand_f1 = _get_version_tag_metric(
            client,
            model_name,
            model_version,
            "f1",
        )

    ### Retrieve current Production baseline
    prod_v, prod_run_id, prod_f1 = _get_prod_baseline(client, model_name)

    ### Determine baseline threshold
    baseline = prod_f1 if prod_f1 is not None else min_f1

    ### Evaluate gate condition
    passed = (cand_f1 is not None) and (cand_f1 >= baseline)

    ### Collect evaluation details for reporting
    details = {
        "candidate_f1": cand_f1,
        "baseline_f1": baseline,
        "production_version": prod_v,
        "production_run_id": prod_run_id,
        "used_fallback_min_f1": prod_f1 is None,
    }

    return GateResult(passed=passed, details=details)

### Helper : gate_api_smoke_schema_latency()
def gate_api_smoke_schema_latency(staging_url: str) -> GateResult:
    """
    Executes API-level validation gates against the staging environment.

    This gate performs:
        - GET /health service availability
        - GET /options feature availability
        - GET /openapi.json schema validation for POST /predict
        - POST /predict multiple times to measure latency (p95)

    The gate fails if:
        - OpenAPI schema cannot be resolved
        - latency p95 exceeds GATE_MAX_P95_MS threshold

    :param:
        staging_url str: base URL of the staging API

    :return:
        GateResult: pass/fail status with detailed evaluation metadata
    """
    ### Normalize staging URL
    staging_url = staging_url.rstrip("/")

    ### Retrieve latency gate configuration from environment
    max_p95_ms = _get_env_float("GATE_MAX_P95_MS", 800.0)
    n = _get_env_int("GATE_LATENCY_N", 15)

    details: Dict[str, Any] = {"staging_url": staging_url}

    ### Health check
    health = _api_get_json(f"{staging_url}/health")
    details["health"] = health

    ### Options endpoint validation
    options = _api_get_json(f"{staging_url}/options")
    details["options_keys"] = (
        list(options.keys()) if isinstance(options, dict)
        else type(options).__name__
    )

    ### OpenAPI schema extraction
    openapi = _api_get_json(f"{staging_url}/openapi.json")
    schema = _extract_predict_schema_from_openapi(openapi)

    ### Fail early if schema is missing
    if not schema:
        return GateResult(
            False,
            {"error": "OpenAPI schema for POST /predict not found", **details},
        )

    ### Resolve schema references if needed
    schema = _resolve_schema_ref(openapi, schema)

    ### Build a valid payload dynamically from schema and options
    payload = _build_payload_from_schema(schema, options)
    details["predict_payload_built"] = payload

    ### Predict calls and latency measurement
    lat_s: list[float] = []
    last_resp: Any = None

    for _ in range(n):
        t0 = time.perf_counter()
        last_resp = _api_post_json(f"{staging_url}/predict", payload)
        lat_s.append(time.perf_counter() - t0)

    ### Compute latency p95 in milliseconds
    p95 = _p95_ms(lat_s)
    details["latency_p95_ms"] = p95
    details["last_predict_response"] = last_resp

    ### Latency gate evaluation ---
    passed = True

    if p95 > max_p95_ms:
        passed = False
        details["latency_gate_failed"] = True
        details["latency_max_p95_ms"] = max_p95_ms

    return GateResult(passed=passed, details=details)

### Helper : promote_to_production()
def promote_to_production(
    client: MlflowClient,
    model_name: str,
    model_version: str
) -> None:
    """
    Promotes a model version to Production in the mlflow registry.

    The promotion includes:
        - transitioning the model to the production stage
        - optionally setting the "production" alias if supported

    :param:
        client MlflowClient: initialized mlflow client
        model_name str: registered model name
        model_version str: model version to promote

    :return:
        None
    """
    ### Transition model version to production stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )

    ### Assign production alias
    if hasattr(client, "set_registered_model_alias"):
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=model_version,
            )
        except Exception:
            ### Ignore alias errors to avoid blocking promotion
            pass

### ----------------------------- Functions ----------------------------- ###
### Function : main()
def main() -> None:
    """
    Executes automated model promotion workflow with quality gates.

    This entrypoint:
        - evaluates metric threshold gate
        - evaluates API smoke/schema/latency gate
        - tags model version with gate results
        - promotes model to Production if all gates pass
        - exits with appropriate status code for CI

    :param:
        None

    :return:
        None
    """
    ### Parse cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "chocolate_sales_logreg"))
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--run-id", default=os.getenv("RUN_ID"))
    parser.add_argument("--staging-url", default=os.getenv("STAGING_API_URL"))
    parser.add_argument("--force-promote", action="store_true", help="Bypass gates and promote (one-time init).")
    parser.add_argument("--tag-prefix", default="gate.")
    args = parser.parse_args()

    ### Initialize mlflow client
    client = _mlflow_client()

    ### Gate 1 : Metric threshold
    metric_gate = gate_metric_threshold(
        client,
        args.model_name,
        args.model_version,
        args.run_id,
    )

    ### Gate 2 : API smoke and schema and latency
    if args.staging_url:
        api_gate = gate_api_smoke_schema_latency(args.staging_url)
    else:
        ### Fail gate if staging URL is missing
        api_gate = GateResult(
            passed=False,
            details={"error": "Missing STAGING_API_URL (cannot run API smoke/schema/latency gate)"},
        )

    ### Determine overall gate result
    overall_pass = args.force_promote or (metric_gate.passed and api_gate.passed)

    ### Tag model version with gate results
    tags: Dict[str, Any] = {
        f"{args.tag_prefix}metric_pass": metric_gate.passed,
        f"{args.tag_prefix}candidate_f1": metric_gate.details.get("candidate_f1"),
        f"{args.tag_prefix}baseline_f1": metric_gate.details.get("baseline_f1"),
        f"{args.tag_prefix}used_fallback_min_f1": metric_gate.details.get("used_fallback_min_f1"),
        f"{args.tag_prefix}api_pass": api_gate.passed,
        f"{args.tag_prefix}latency_p95_ms": api_gate.details.get("latency_p95_ms"),
        f"{args.tag_prefix}overall_pass": overall_pass,
        f"{args.tag_prefix}staging_url": args.staging_url or "",
        f"{args.tag_prefix}timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    _tag_version(client, args.model_name, args.model_version, tags)

    ### Attach condensed json details as a tag
    try:
        blob = {
            "metric_gate": {"passed": metric_gate.passed, "details": metric_gate.details},
            "api_gate": {"passed": api_gate.passed, "details": api_gate.details},
        }
        _tag_version(
            client,
            args.model_name,
            args.model_version,
            {f"{args.tag_prefix}details_json": json.dumps(blob)[:5000]},
        )
    except Exception:
        ### Ignore json tagging errors
        pass

    ### Promotion decision
    if overall_pass:
        promote_to_production(client, args.model_name, args.model_version)
        print(f"PROMOTION=PASS model={args.model_name} version={args.model_version}")
        sys.exit(0)

    ### If gates fail exit with non-zero code
    print(f"PROMOTION=FAIL model={args.model_name} version={args.model_version}")
    print("Metric gate:", metric_gate.passed, metric_gate.details)
    print("API gate:", api_gate.passed, list(api_gate.details.keys()))
    sys.exit(1)


if __name__ == "__main__":
    main()
