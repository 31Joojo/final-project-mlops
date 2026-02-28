"""
tests/unit/test_amount_cleaning.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
import pandas as pd

from ml.training.train import _clean_amount_to_float

### ------------------------------- Tests ------------------------------- ###
### Test : test_clean_amount_to_float()
def test_clean_amount_to_float():
    """
    Verifies that currency-formatted strings are correctly cleaned and converted to float values.

    :return:
        None: asserts proper numeric conversion and NaN handling
    """
    ### Create sample monetary strings with various formats
    s = pd.Series(["$1,234", "$0", "  $5,320  ", "", "nan"])

    ### Apply cleaning function
    out = _clean_amount_to_float(s)

    ### Assert proper numeric conversion
    assert out.iloc[0] == 1234.0
    assert out.iloc[1] == 0.0
    assert out.iloc[2] == 5320.0

    ### Assert empty and nan strings become NaN
    assert pd.isna(out.iloc[3])
    assert pd.isna(out.iloc[4])
