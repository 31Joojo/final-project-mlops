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

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    DVC pull with DagsHub storage using HTTP basic auth.
    Expects DAGSHUB_USERNAME + DAGSHUB_TOKEN in env.
    Uses --local so nothing is committed.
    """
    ### Retrieve configured DVC remote name
    dvc_remote = os.getenv("DVC_REMOTE", "origin")

    ### Retrieve DagsHub credentials from environment
    user = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    ### Ensure required credentials are present
    if not user or not token:
        raise RuntimeError("Missing DAGSHUB_USERNAME and/or DAGSHUB_TOKEN environment variables.")

    ### Prepare environment variables for non-interactive execution
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["DVC_NO_ANALYTICS"] = "1"

    ### Helper to execute dvc commands locally with modified environment
    ### Helper : run_local()
    def run_local(args: list[str]) -> None:
        subprocess.run(args, check=True, text=True, env=env)

    ### Configure remote authentication
    ### Try setting auth=basic then fallback to user/password only
    try:
        run_local(["dvc", "remote", "modify", dvc_remote, "--local", "auth", "basic"])
    except subprocess.CalledProcessError:
        pass

    run_local(["dvc", "remote", "modify", dvc_remote, "--local", "user", user])
    run_local(["dvc", "remote", "modify", dvc_remote, "--local", "password", token])
    run_local(["dvc", "remote", "modify", dvc_remote, "--local", "ask_password", "false"])

    ### Pull dataset
    run_local(["dvc", "pull", data_path, "-q"])

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
    label_quantile: float,
    set_stage: str,
    set_alias: bool,
) -> TrainOutputs:
    """
    Executes the full training, logging, registration and promotion workflow.

    Steps:
        - load dataset
        - detect schema
        - build features and labels
        - train Logistic Regression pipeline
        - log params, metrics, and tags to mlflow
        - register model version
        - transition model to target stage
        - optionally set alias
        - tag model version for traceability

    :param:
        model_name str: registered mlflow model name
        experiment_name str: mlflow experiment name
        data_path str: path to training dataset
        label_quantile float: quantile used to define positive class
        set_stage str: stage to transition the model to
        set_alias bool: whether to set registry alias

    :return:
        TrainOutputs: run id, model version and computed metrics
    """
    ### Load dataset
    df = pd.read_csv(data_path)

    ### Detect date and target columns automatically
    date_col, target_col = _detect_columns(df)

    ### Build feature matrix X and binary target y
    X, y, meta = _build_training_frame(df, date_col, target_col, label_quantile)

    ### Split dataset into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ### Build preprocessing pipeline
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), meta["num_cols"]),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), meta["cat_cols"]),
        ],
        remainder="drop",
    )

    ### Define classifier
    clf = LogisticRegression(max_iter=2000)

    ### Combine preprocessing and classifier
    pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])

    ### Configure mlflow experiment
    mlflow.set_experiment(experiment_name)

    ### Collect traceability information
    git_sha = _git_sha()
    dvc_md5 = _read_dvc_md5(Path(f"{data_path}.dvc"))

    ### Start mlflow
    with mlflow.start_run() as run:
        ### Train model
        pipe.fit(X_train, y_train)

        ### Evaluate on test set
        y_pred = pipe.predict(X_test)

        ### Compute evaluation metrics
        metrics = {
            "f1": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        }

        ### Log training parameters
        mlflow.log_params(
            {
                "model_family": "logreg",
                "label_quantile": label_quantile,
                "label_threshold": meta["label_threshold"],
                "cat_cols": ",".join(meta["cat_cols"]),
                "num_cols": ",".join(meta["num_cols"]),
            }
        )

        ### Log evaluation metrics
        mlflow.log_metrics(metrics)

        ### Add traceability tags
        mlflow.set_tag("git_sha", git_sha)
        mlflow.set_tag("data_path", data_path)
        if dvc_md5:
            mlflow.set_tag("dvc_md5", dvc_md5)

        ### Prepare model signature for deployment validation
        input_example = X_train.head(5)
        try:
            from mlflow.models.signature import infer_signature

            signature = infer_signature(input_example, pipe.predict(input_example))
        except Exception:
            signature = None

        ### Log model artifact and register new model version
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

        ### Store mlflow run id
        run_id = run.info.run_id

    ### Resolve model version from registry
    client = MlflowClient()
    model_version = None

    ### Poll registry because it can be eventually consistent
    for _ in range(30):
        versions = client.search_model_versions(f"name = '{model_name}'")

        ### Identify version linked to this run
        for v in versions:
            if getattr(v, "run_id", None) == run_id:
                model_version = str(v.version)
                break
        if model_version:
            break
        time.sleep(2)

    ### Fail if version could not be resolved
    if not model_version:
        raise RuntimeError("Could not resolve model version after registration (search_model_versions returned none).")

    ### Stage transition
    if set_stage:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=set_stage,
            archive_existing_versions=True,
        )

    ### Set alias if mlflow version doesn't support aliases => we ignore
    if set_alias and hasattr(client, "set_registered_model_alias"):
        try:
            ### Common alias names staging or production
            client.set_registered_model_alias(name=model_name, alias="staging", version=model_version)
        except Exception:
            ### Ignore alias errors
            pass

    ### Tag model version for traceability
    client.set_model_version_tag(model_name, model_version, "git_sha", git_sha)
    client.set_model_version_tag(model_name, model_version, "data_path", data_path)

    if dvc_md5:
        client.set_model_version_tag(model_name, model_version, "dvc_md5", dvc_md5)
    client.set_model_version_tag(model_name, model_version, "run_id", run_id)

    ### Attach metric values as tags
    for k, v in metrics.items():
        client.set_model_version_tag(model_name, model_version, f"metric.{k}", f"{v:.6f}")

    ### Return structured outputs
    return TrainOutputs(run_id=run_id, model_version=model_version, metrics=metrics)

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
    parser.add_argument("--label-quantile", type=float, default=float(os.getenv("LABEL_QUANTILE", "0.5")))
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
        label_quantile=args.label_quantile,
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
