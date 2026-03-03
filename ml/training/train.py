"""
ml/training/train.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

import argparse
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .utils import get_dvc_data_rev, get_env, get_git_commit

### ------------------------------ Helpers ------------------------------ ###
### Helper : _clean_amount_to_float()
def _clean_amount_to_float(s: pd.Series) -> pd.Series:
    """
    Cleans and converts currency-formatted strings into float values.

    :param:
        s pd.Series: series containing monetary values as strings

    :return:
        pd.Series: cleaned numeric series with float dtype
    """
    return (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )

### Helper : load_chocolate_sales()
def load_chocolate_sales(path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the chocolate sales dataset with basic feature engineering.

    :param:
        path str: path to the CSV dataset

    :return:
        pd.DataFrame: cleaned dataframe ready for model training
    """
    ### Read dataset from CSV file
    df = pd.read_csv(path)

    ### Ensure consistent column formatting
    df.columns = [c.strip() for c in df.columns]

    ### Validate required schema
    required = {"Sales Person", "Country", "Product", "Date", "Amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

    ### Clean monetary column
    df["Amount"] = _clean_amount_to_float(df["Amount"])

    ### Boxes Shipped added
    if "Boxes Shipped" not in df.columns:
        df["Boxes Shipped"] = np.nan

    ### Ensure Boxes Shipped is numeric for modeling
    df["Boxes Shipped"] = pd.to_numeric(df["Boxes Shipped"], errors="coerce")

    ### Date parsing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    ### Extract temporal features
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["quarter"] = df["Date"].dt.quarter
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype("float64")

    ### Enforce float dtype consistency
    for c in ["Boxes Shipped", "year", "month", "dayofweek", "quarter"]:
        df[c] = df[c].astype("float64")

    return df

### ----------------------------- Fonctions ----------------------------- ###
### Function : build_pipeline()
def build_pipeline(X: pd.DataFrame, random_state: int, C: float, penalty: str,
    class_weight: Optional[str], onehot_min_frequency: Optional[float], onehot_max_categories: Optional[int],) -> Pipeline:
    """
    Builds a preprocessing and classification pipeline using scikit-learn.

    :param:
        X pd.DataFrame: feature dataframe used to determine column types
        random_state int: random seed for reproducibility
        C float: inverse regularization strength for LogisticRegression
        penalty str: regularization type
        class_weight Optional[str]: class balancing strategy
        onehot_min_frequency Optional[float]: minimum category frequency for grouping
        onehot_max_categories Optional[int]: maximum categories per feature

    :return:
        Pipeline: configured preprocessing and LogisticRegression pipeline
    """
    ### Detect categorical and numerical columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    ### Median imputation for numeric features
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    ohe_kwargs = {"handle_unknown": "ignore"}
    if onehot_min_frequency is not None:
        ohe_kwargs["min_frequency"] = onehot_min_frequency
    if onehot_max_categories is not None:
        ohe_kwargs["max_categories"] = onehot_max_categories
    ohe_kwargs["sparse_output"] = True

    ### Impute and encode categorical features
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**ohe_kwargs)),
        ]
    )

    ### Combining preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    ### Configure Logistic Regression classifier
    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=random_state,
        C=C,
        penalty=penalty,
        class_weight=class_weight,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

### Function : main()
def main() -> None:
    """
    Executes the end-to-end configurable training workflow including
    data preprocessing, feature engineering, model training, evaluation,
    and MLflow experiment tracking with model registration.
    """
    ### Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/raw/chocolate_sales.csv")
    parser.add_argument("--model-name", type=str, default="chocolate_sales_logreg")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--label-strategy",
        type=str,
        default="median",
        choices=["median", "p75", "p90"],
        help="How to binarize Amount into high/low",
    )

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Use balanced if target becomes imbalanced",
    )
    parser.add_argument(
        "--onehot-min-frequency",
        type=float,
        default=0.0,
        help="Group rare categories. Use 0 to disable",
    )
    parser.add_argument(
        "--onehot-max-categories",
        type=int,
        default=0,
        help="Limit categories per feature. Use 0 to disable",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="candidate",
        help="Tag to identify runs",
    )

    parser.add_argument("--tracking-uri", type=str, default="")
    parser.add_argument("--no-register", action="store_true")

    args = parser.parse_args()

    experiment_name = get_env("MLFLOW_EXPERIMENT_NAME", "train-chocolate-sales") or "train-chocolate-sales"

    ### Configure mlflow tracking URI when provided
    tracking_uri = get_env("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    registry_uri = get_env("MLFLOW_REGISTRY_URI")
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    mlflow.set_experiment(experiment_name)

    df = load_chocolate_sales(args.data_path)

    ### Drop rows with missing Amount
    df = df.dropna(subset=["Amount"]).reset_index(drop=True)

    ### Create binary target from Amount column
    amount = df["Amount"]
    if args.label_strategy == "median":
        thr = float(np.nanmedian(amount))
    else:
        thr = float(np.nanquantile(amount, 0.75))

    df["high_amount"] = (df["Amount"] >= thr).astype(int)

    ### Select training features
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

    ### Split dataset into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    class_weight = None if args.class_weight == "none" else "balanced"
    onehot_min_frequency = None if args.onehot_min_frequency <= 0 else args.onehot_min_frequency
    onehot_max_categories = None if args.onehot_max_categories <= 0 else args.onehot_max_categories

    ### Build pipeline
    pipe = build_pipeline(
        X_train,
        random_state=args.random_state,
        C=args.C,
        penalty=args.penalty,
        class_weight=class_weight,
        onehot_min_frequency=onehot_min_frequency,
        onehot_max_categories=onehot_max_categories,
    )

    ### Start mlflow run context
    with mlflow.start_run(run_name=args.model_name):
        ### Traceability tags
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_data_rev", get_dvc_data_rev(args.data_path))
        mlflow.set_tag("dataset_path", args.data_path)
        mlflow.set_tag("run_tag", args.run_tag)

        ### Log training parameters
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 2000,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "label_strategy": args.label_strategy,
                "amount_threshold": thr,
                "target": "high_amount",
                "C": args.C,
                "penalty": args.penalty,
                "class_weight": args.class_weight,
                "onehot_min_frequency": args.onehot_min_frequency,
                "onehot_max_categories": args.onehot_max_categories,
            }
        )

        ### Train
        pipe.fit(X_train, y_train)

        ### Compute evaluation metrics
        y_pred = pipe.predict(X_val)

        ### Log metrics to mlflow
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1": float(f1_score(y_val, y_pred)),
        }
        y_proba = pipe.predict_proba(X_val)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba))
        mlflow.log_metrics(metrics)

        ### Infer model signature for deployment compatibility
        signature = infer_signature(X_val, y_proba)
        input_example = X_train.head(5)

        ### Log and register model in mlflow model registry
        if args.no_register:
            mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                signature=signature,
                input_example=input_example,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                registered_model_name=args.model_name,
                signature=signature,
                input_example=input_example,
            )

        run_id = mlflow.active_run().info.run_id
        print("MLflow run logged and model registered")
        print(f"experiment: {experiment_name}")
        print(f"run_id: {run_id}")
        print(f"registered_model_name: {args.model_name}")
        print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
