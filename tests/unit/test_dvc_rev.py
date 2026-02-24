"""
tests/unit/test_dvc_rev.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from pathlib import Path

from ml.training.utils import get_dvc_data_rev

### ------------------------------- Tests ------------------------------- ###
### Test : test_get_dvc_data_rev_untracked()
def test_get_dvc_data_rev_untracked(tmp_path: Path):
    """
    Verifies that get_dvc_data_rev returns "untracked" when no .dvc file exists.

    :param:
        tmp_path Path: temporary directory fixture provided by pytest

    :return:
        None: asserts untracked datasets are correctly identified
    """
    ### Create temporary CSV file without DVC tracking
    data_path = tmp_path / "data.csv"
    data_path.write_text("a,b\n1,2\n")

    ### Expect dataset to be marked as untracked
    assert get_dvc_data_rev(str(data_path)) == "untracked"

### Test : test_get_dvc_data_rev_reads_md5()
def test_get_dvc_data_rev_reads_md5(tmp_path: Path):
    """
    Verifies that get_dvc_data_rev correctly extracts the md5 hash from a .dvc file.

    :param:
        tmp_path Path: temporary directory fixture provided by pytest

    :return:
        None: asserts correct DVC revision string is returned
    """
    ### Create temporary CSV file
    data_path = tmp_path / "data.csv"
    data_path.write_text("a,b\n1,2\n")

    ### Create corresponding .dvc file with md5 hash
    dvc_file = tmp_path / "data.csv.dvc"
    dvc_file.write_text(
        """
outs:
  - md5: 1234567890abcdef1234567890abcdef
    size: 12
    path: data.csv
"""
    )

    ### Retrieve revision string
    rev = get_dvc_data_rev(str(data_path))

    ### Assert correct md5-based revision format
    assert rev == "md5:1234567890abcdef1234567890abcdef"
