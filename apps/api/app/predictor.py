"""
apps/api/app/predictor.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

import math
from typing import Tuple

from .schema import PredictRequest

### ----------------------------- Functions ----------------------------- ###
### Function : sigmoid()
def sigmoid(x: float) -> float:
    """
    Computes the sigmoid function used to map a real-valued score to a probability.

    :param:
        x float: real-valued input score

    :returns:
        float: probability value in the range (0, 1)
    """
    return 1.0 / (1.0 + math.exp(-x))

### Function : dummy_classifier()
def dummy_classifier(req: PredictRequest) -> Tuple[int, float]:
    """
    Performs a deterministic placeholder classification based on transaction revenue.

    :param:
        req PredictRequest: validated input features used to compute the revenue score

    :returns:
        tuple[int, float]: predicted class label and associated probability score
    """
    ### Compute transaction revenue
    revenue = req.quantity * req.unit_price * (1.0 - req.discount)

    ### Transform revenue into a stable scoring space
    score = math.log1p(revenue) - 8.0

    ### Convert score to probability using sigmoid
    proba = sigmoid(score)

    ### Apply binary decision rule
    pred = 1 if proba >= 0.5 else 0

    return pred, proba
