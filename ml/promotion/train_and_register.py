"""
ml/promotion/train_and_register.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np

from ml.training.train import load_chocolate_sales, build_pipeline
from ml.training.utils import get_git_commit, get_dvc_data_rev, get_env
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


try:
    import yaml
except Exception:
    yaml = None

### ------------------------------ Classes ------------------------------ ###
### Class : TrainOutputs
@dataclass
class TrainOutputs:
    """
    Container for outputs produced by the training and registration pipeline.

    :param:
        run_id str: mlflow run identifier
        model_version str: registered mlflow model version
        metrics Dict[str, float]: evaluation metrics computed during training

    :return:
        TrainOutputs: structured training outputs
    """
    run_id: str
    model_version: str
    metrics: Dict[str, float]

### ------------------------------ Helpers ------------------------------ ###
### Helper : _run()
def _run(cmd: list[str], *, cwd: Optional[str] = None) -> None:
    """
    Executes a shell command and raises an exception if it fails.

    :param:
        cmd list[str]: command and arguments to execute
        cwd Optional[str]: optional working directory

    :return:
        None
    """
    subprocess.run(cmd, cwd=cwd, check=True, text=True)

### Helper : _ensure_dvc_pull()
def _ensure_dvc_pull(data_path: str) -> None:
    """
    Synchronizes the local dataset from DagsHub DVC storage using token-based authentication.

    This function:
        - expects DAGSHUB_TOKEN in the environment
        - configures the DVC "origin" remote locally with S3-compatible credentials
        - pulls the specified dataset using the pip-installed DVC binary

    :param:
        data_path str: path to the dvc-tracked dataset file

    :return:
        None

    :raises:
        RuntimeError: if DAGSHUB_TOKEN is missing
        subprocess.CalledProcessError: if any DVC command fails
    """
    ### Retrieve DagsHub token from environment
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise RuntimeError("Missing DAGSHUB_TOKEN.")

    ### Prepare non-interactive execution environment
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["DVC_NO_ANALYTICS"] = "1"

    ### Force usage of pip-installed dvc
    dvc = [sys.executable, "-m", "dvc"]

    ### Reset local dvc config to avoid stale or invalid settings
    cfg_local = Path(".dvc/config.local")
    if cfg_local.exists():
        cfg_local.unlink()

    ### Configure dagshub remote credentials locally
    subprocess.run(
        dvc + ["remote", "modify", "origin", "--local", "access_key_id", token],
        check=True,
        text=True,
        env=env,
    )

    subprocess.run(
        dvc + ["remote", "modify", "origin", "--local", "secret_access_key", token],
        check=True,
        text=True,
        env=env,
    )

    ### Pull dataset from remote
    subprocess.run(
        dvc + ["pull", data_path, "-v"],
        check=True,
        text=True,
        env=env,
    )

### Helper : _git_sha()
def _git_sha() -> str:
    """
    Retrieves the current git commit sha for traceability.

    :param:
        None

    :return:
        str: commit SHA or unknown if unavailable
    """
    ### Try retrieving sha from GitHub actions environment
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha

    ### Fallback: retrieve sha from local git repository
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:

        ### Return fallback value if git is unavailable
        return "unknown"

### Helper : _read_dvc_md5()
def _read_dvc_md5(dvc_file: Path) -> Optional[str]:
    """
    Extracts the dvc md5 checksum from a .dvc file for data traceability.

    :param:
        dvc_file Path: path to the .dvc file

    :return:
        Optional[str]: md5 hash if found
    """
    ### Return None if .dvc file does not exist
    if not dvc_file.exists():
        return None

    ### Read raw file content
    txt = dvc_file.read_text(encoding="utf-8")

    ### Try parsing using yaml if available
    if yaml is not None:
        try:
            obj = yaml.safe_load(txt)
            outs = obj.get("outs", [])
            if outs and isinstance(outs[0], dict):
                return outs[0].get("md5")
        except Exception:
            ### Ignore YAML parsing errors and fallback to manual parsing
            pass

    ### Fallback: naive line-by-line parsing
    for line in txt.splitlines():
        if line.strip().startswith("md5:"):
            return line.split("md5:", 1)[1].strip()

    ### Return None if no md5 found
    return None

### Helper : _detect_columns()
def _detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detects the date column and target column from a dataset using heuristic rules.

    :param:
        df pd.DataFrame: input dataset

    :return:
        Tuple[str, str]: detected (date_column_name, target_column_name)

    :raises:
        RuntimeError: if no suitable date or numeric target column can be detected
    """
    ### Build case-insensitive column lookup
    cols_lower = {c.lower(): c for c in df.columns}

    ### Try common explicit date column names
    for key in ["date", "sale date", "order date"]:
        if key in cols_lower:
            date_col = cols_lower[key]
            break
    else:
        ### Fallback: any column containing the word date
        candidates = [c for c in df.columns if "date" in c.lower()]
        date_col = candidates[0] if candidates else ""

    ### Try common business-related target names
    for key in ["amount", "sale amount", "revenue", "sales", "profit"]:
        if key in cols_lower:
            target_col = cols_lower[key]
            break
    else:
        ### Fallback: select the last numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise RuntimeError(
                "Could not detect a numeric target column (Amount/Revenue/etc.)."
            )
        target_col = num_cols[-1]

    ### Ensure date column was detected
    if not date_col:
        raise RuntimeError(
            "Could not detect a date column (expected something like 'Date' or 'Sale Date')."
        )

    return date_col, target_col

### Helper : _build_training_fram()
def _build_training_frame(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    label_quantile: float,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Builds the training feature matrix and binary target variable.

    The label is defined as:
        y = 1 if target >= quantile(target), else 0

    :param:
        df pd.DataFrame: input dataset
        date_col str: detected date column name
        target_col str: detected target column name
        label_quantile float: quantile used to define positive class threshold

    :return:
        Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
            - X: feature dataframe
            - y: binary target series
            - meta: metadata dictionary describing training configuration

    :raises:
        RuntimeError: if target column is missing
    """
    ### Work on a copy to avoid mutating original dataframe
    df = df.copy()

    ### Parse date column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    ### Extract temporal features
    df["year"] = df[date_col].dt.year.fillna(1970).astype(int)
    df["month"] = df[date_col].dt.month.fillna(1).astype(int)
    df["dayofweek"] = df[date_col].dt.dayofweek.fillna(0).astype(int)

    ### Expected categorical and numerical columns
    expected_cat = ["Sales Person", "Country", "Product"]
    expected_num = ["Boxes Shipped", "year", "month", "dayofweek"]

    ### Select available categorical and numerical columns
    cat_cols = [c for c in expected_cat if c in df.columns]
    num_cols = [c for c in expected_num if c in df.columns]

    ### Fallback: detect object columns
    if not cat_cols:
        cat_cols = [
            c for c in df.columns
            if df[c].dtype == "object" and c not in [date_col, target_col]
        ]

    ### Fallback: detect numeric columns
    if not num_cols:
        num_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != target_col
        ]

    ### Ensure target column exists
    if target_col not in df.columns:
        raise RuntimeError(
            f"Target column '{target_col}' not found in dataframe."
        )

    ### Build binary target based on quantile threshold
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    thr = float(y_raw.quantile(label_quantile))
    y = (y_raw >= thr).astype(int)

    ### Construct feature matrix
    X = df[cat_cols + num_cols].copy()

    ### Store metadata for traceability
    meta = {
        "date_col": date_col,
        "target_col": target_col,
        "label_quantile": label_quantile,
        "label_threshold": thr,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }

    return X, y, meta

### Helper : _train_log_register()
def _train_log_register(
    model_name: str,
    experiment_name: str,
    data_path: str,
    label_strategy: str,
    set_stage: str,
    set_alias: bool,
) -> TrainOutputs:
    """
    Executes end-to-end candidate training and registration workflow.

    This function:
        - configures mlflow tracking and registry from environment
        - loads and preprocesses the dataset
        - builds training labels based on selected strategy
        - trains a Logistic Regression pipeline
        - logs metrics, params, and traceability metadata
        - registers a new model version
        - transitions it to the desired stage
        - optionally assigns a registry alias

    :param:
        model_name str: registered mlflow model name
        experiment_name str: mlflow experiment name
        data_path str: path to training dataset
        label_strategy str: labeling strategy ("median", "p75", "p90")
        set_stage str: stage to transition the model to
        set_alias bool: whether to assign a registry alias

    :return:
        TrainOutputs: run id, model version, and computed metrics
    """

    ### Mlflow configuration from environment
    tracking_uri = get_env("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    registry_uri = get_env("MLFLOW_REGISTRY_URI")
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    exp_name = get_env("MLFLOW_EXPERIMENT_NAME", experiment_name) or experiment_name
    mlflow.set_experiment(exp_name)

    ### Load and preprocess dataset
    df = load_chocolate_sales(data_path)
    df = df.dropna(subset=["Amount"]).reset_index(drop=True)

    ### Label creation
    amount = df["Amount"]
    if label_strategy == "median":
        thr = float(np.nanmedian(amount))
    elif label_strategy == "p75":
        thr = float(np.nanquantile(amount, 0.75))
    elif label_strategy == "p90":
        thr = float(np.nanquantile(amount, 0.90))
    else:
        raise ValueError(f"Unknown label_strategy={label_strategy}")

    df["high_amount"] = (df["Amount"] >= thr).astype(int)

    feature_cols = [
        "Sales Person",
        "Country",
        "Product",
        "Boxes Shipped",
        "year",
        "month",
        "dayofweek",
        ## "quarter",
        ## "weekofyear",
    ]

    X = df[feature_cols].copy()
    y = df["high_amount"].copy()

    ### Guardrail: ensure binary classification is valid
    if y.nunique() < 2:
        raise RuntimeError(
            f"Training label has only one class "
            f"(distribution={y.value_counts().to_dict()})."
        )

    ### Train/validation split
    test_size = float(get_env("TRAIN_TEST_SIZE", "0.2"))
    random_state = int(get_env("TRAIN_RANDOM_STATE", "42"))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    ### Build pipeline
    class_weight_env = get_env("TRAIN_CLASS_WEIGHT", "none")
    class_weight = None if class_weight_env == "none" else "balanced"

    C = float(get_env("TRAIN_C", "1.0"))
    penalty = get_env("TRAIN_PENALTY", "l2")

    onehot_min_frequency = get_env("TRAIN_ONEHOT_MIN_FREQUENCY", "")
    onehot_min_frequency = None if onehot_min_frequency in ("", "0", None) else float(onehot_min_frequency)

    onehot_max_categories = get_env("TRAIN_ONEHOT_MAX_CATEGORIES", "")
    onehot_max_categories = None if onehot_max_categories in ("", "0", None) else int(onehot_max_categories)

    pipe = build_pipeline(
        X_train,
        random_state=random_state,
        C=C,
        penalty=penalty,
        class_weight=class_weight_env if class_weight else None,
        onehot_min_frequency=onehot_min_frequency,
        onehot_max_categories=onehot_max_categories,
    )

    ### Training and mlflow logging
    with mlflow.start_run(run_name=model_name) as run:

        ### Traceability tags
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_data_rev", get_dvc_data_rev(data_path))
        mlflow.set_tag("dataset_path", data_path)
        mlflow.set_tag("run_tag", "candidate")

        ### Log hyperparameters
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 1000,
                "test_size": test_size,
                "random_state": random_state,
                "label_strategy": label_strategy,
                "amount_threshold": thr,
                "target": "high_amount",
                "C": C,
                "penalty": penalty,
                "class_weight": class_weight_env,
                "onehot_min_frequency": onehot_min_frequency or 0.0,
                "onehot_max_categories": onehot_max_categories or 0,
            }
        )

        ### Train model
        pipe.fit(X_train, y_train)

        ### Evaluate model
        y_pred = pipe.predict(X_val)
        y_proba = pipe.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1": float(f1_score(y_val, y_pred)),
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
        }

        mlflow.log_metrics(metrics)

        ### Log model artifact and register version
        signature = infer_signature(X_val, y_proba)
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

        run_id = run.info.run_id

    ### Resolve registered model version
    client = MlflowClient()
    model_version = None

    for _ in range(30):
        versions = client.search_model_versions(f"name = '{model_name}'")
        for v in versions:
            if getattr(v, "run_id", None) == run_id:
                model_version = str(v.version)
                break
        if model_version:
            break
        time.sleep(2)

    if not model_version:
        raise RuntimeError("Could not resolve model version after registration.")

    ### Stage transition
    if set_stage:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=set_stage,
            archive_existing_versions=True,
        )

    ### Optional alias
    if set_alias and hasattr(client, "set_registered_model_alias"):
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="staging",
                version=model_version,
            )
        except Exception:
            pass

    ### Tag version with metrics for gates
    client.set_model_version_tag(model_name, model_version, "run_id", run_id)

    for k, v in metrics.items():
        client.set_model_version_tag(
            model_name,
            model_version,
            f"metric.{k}",
            f"{v:.6f}",
        )

    return TrainOutputs(
        run_id=run_id,
        model_version=model_version,
        metrics=metrics,
    )

### Helper : _write_github_outputs()
def _write_github_outputs(run_id: str, model_version: str) -> None:
    """
    Writes training outputs to the GitHub Actions output file.

    :param:
        run_id str: mlflow run identifier
        model_version str: registered mlflow model version

    :return:
        None
    """
    ### Retrieve GitHub actions output file path
    out_path = os.getenv("GITHUB_OUTPUT")

    ### If not running inside GitHub actions we do nothing
    if not out_path:
        return

    ### Append outputs in key=value format for workflow consumption
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"run_id={run_id}\n")
        f.write(f"model_version={model_version}\n")

### ----------------------------- Functions ----------------------------- ###
### Function : main()
def main() -> None:
    """
    Entry point for the training and promotion pipeline.

    Workflow:
        1. Pull dataset from DVC remote
        2. Train model and register new version in mLflow
        3. Persist outputs for CI/CD usage
        4. Export GitHub Actions outputs

    :param:
        None

    :return:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "chocolate_sales_logreg"))
    parser.add_argument("--experiment-name", default=os.getenv("MLFLOW_EXPERIMENT", "chocolate-sales"))
    parser.add_argument("--data-path", default="data/raw/chocolate_sales.csv")
    parser.add_argument("--label-strategy", default=os.getenv("LABEL_STRATEGY", "median"), choices=["median", "p75", "p90"])
    parser.add_argument("--set-stage", default=os.getenv("CANDIDATE_STAGE", "Staging"))
    parser.add_argument("--set-alias", action="store_true", default=os.getenv("SET_ALIAS", "1") == "1")
    parser.add_argument("--output-json", default=os.getenv("PROMOTION_OUTPUT_JSON", "ml/promotion/outputs.json"))
    args = parser.parse_args()

    ### DVC pull
    _ensure_dvc_pull(args.data_path)

    ### Train and register
    outputs = _train_log_register(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        label_strategy=args.label_strategy,
        set_stage=args.set_stage,
        set_alias=args.set_alias,
    )

    ### Persist outputs for continuous integration
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(
            {"run_id": outputs.run_id, "model_version": outputs.model_version, "metrics": outputs.metrics},
            indent=2,
        ),
        encoding="utf-8",
    )

    #### Print and GitHub actions outputs
    print(f"RUN_ID={outputs.run_id}")
    print(f"MODEL_VERSION={outputs.model_version}")
    _write_github_outputs(outputs.run_id, outputs.model_version)


if __name__ == "__main__":
    main()
