"""
tests/integration/test_training_smoke.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import os
import subprocess
from pathlib import Path

### ------------------------------- Tests ------------------------------- ###
### Test : test_training_smoke_local_mlflow()
def test_training_smoke_local_mlflow(tmp_path: Path):
    """
    Verifies that the training script runs successfully with a minimal dataset
    and logs results to a local mlflow tracking directory.

    :param:
        tmp_path Path: temporary directory fixture used for dataset and tracking storage

    :return:
        None: asserts training process completes without errors and creates mlflow artifacts
    """
    ### Create mini dataset
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "chocolate_sales.csv"
    csv_path.write_text(
        "Sales Person,Country,Product,Date,Amount,Boxes Shipped\n"
        "Alice,FR,Dark,2024-01-01,$100,10\n"
        "Bob,FR,Milk,2024-01-02,$200,20\n"
        "Alice,US,Dark,2024-01-03,$150,15\n"
        "Cara,UK,White,2024-01-04,$300,30\n"
        "Bob,US,Milk,2024-01-05,$120,12\n"
        "Cara,FR,White,2024-01-06,$250,25\n"
    )

    ### Configure local mlflow tracking URI
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file://{tracking_dir}"

    ### Load environment variables
    env = os.environ.copy()
    env["MLFLOW_EXPERIMENT_NAME"] = "test-train-chocolate"

    ### Execute training script
    cmd = [
        "python",
        "-m",
        "ml.training.train",
        "--data-path",
        str(csv_path),
        "--model-name",
        "chocolate_sales_logreg_test",
        "--test-size",
        "0.3",
        "--random-state",
        "42",
        "--tracking-uri",
        tracking_uri,
        "--no-register",
        "--run-tag",
        "test",
    ]

    ### Run command and capture output
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)

    ### Ensure training script completed successfully
    assert res.returncode == 0, res.stderr

    ### Ensure MLflow tracking directory was created
    assert tracking_dir.exists()
