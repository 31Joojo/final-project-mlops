"""
ml/training/utils.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import yaml

### ------------------------------ Helpers ------------------------------ ###
### Helper : run_cmd()
def run_cmd(cmd: list[str]) -> str:
    """
    Executes a shell command and returns its output as a string.

    :param:
        cmd list[str]: command and arguments to execute

    :return:
        str: command output or "unknown" if execution fails
    """
    ### Execute shell command and return output or fallback
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return out
    except Exception:
        return "unknown"

### Helper : get_git_commit()
def get_git_commit() -> str:
    """
    Retrieves the current Git commit hash for reproducibility tracking.

    :return:
        str: Git commit SHA or "unknown" if unavailable
    """
    ### Retrieve current Git commit hash
    sha = run_cmd(["git", "rev-parse", "HEAD"])
    if sha != "unknown":
        return sha

    ### Fallback to GitHub actions SHA if available
    return os.getenv("GITHUB_SHA", "unknown")

### Helper : get_dvc_data_rev()
def get_dvc_data_rev(data_path: str) -> str:
    """
    Retrieves the DVC-tracked data version hash from the corresponding .dvc file.

    :param:
        data_path str: path to the tracked dataset

    :return:
        str: data version identifier or fallback value
    """
    ### Locate corresponding .dvc file
    dvc_file = Path(f"{data_path}.dvc")
    if not dvc_file.exists():
        return "untracked"

    ### Parse yaml content of .dvc file
    try:
        doc = yaml.safe_load(dvc_file.read_text())
    except Exception:
        return "unknown"

    ### Extract first output entry
    outs = doc.get("outs", []) if isinstance(doc, dict) else []
    if not outs:
        return "unknown"

    o = outs[0]

    ### Return available hash type
    if "md5" in o:
        return f"md5:{o['md5']}"
    if "etag" in o:
        return f"etag:{o['etag']}"
    if "checksum" in o:
        return f"checksum:{o['checksum']}"
    return "unknown"

### Helper : get_env()
def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves an environment variable value with a default fallback.

    :param:
        name str: environment variable name
        default Optional[str]: fallback value if variable is unset or empty

    :return:
        Optional[str]: resolved environment variable value
    """
    ### Retrieve environment variable with empty-string protection
    v = os.getenv(name)
    return v if v not in (None, "") else default
