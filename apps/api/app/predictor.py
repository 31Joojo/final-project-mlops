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
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from .model_loader import LoadedModel, ModelMeta
from .schema import PredictRequest

logger = logging.getLogger(__name__)

### ------------------------------ Classes ------------------------------ ###
### Class : ModelNotLoadedError
class ModelNotLoadedError(RuntimeError):
    """
    Raised when /predict is called but no model is loaded.
    """
    pass

### Class : PredictOutput
@dataclass
class PredictOutput:
    """
    Represents the output of a prediction request.

    :param:
        prediction int: predicted class label
        probability Optional[float]: predicted probability for positive class

    :return:
        PredictOutput: structured prediction result
    """
    prediction: int
    probability: Optional[float]

### Class : Predictor
class Predictor:
    """
    Manages a loaded mlflow model in memory and performs inference requests.

    :param:
        None

    :return:
        Predictor: model wrapper for runtime predictions
    """
    def __init__(self) -> None:
        """
        Initializes the Predictor without any loaded model.

        :param:
            None

        :return:
            None
        """
        ### Store loaded model in memory
        self._loaded: Optional[LoadedModel] = None

    ### Method : loaded()
    @property
    def loaded(self) -> bool:
        """
        Indicates whether a model is currently loaded in memory.

        :param:
            None

        :return:
            bool: True if a model is loaded
        """
        ### Indicates whether a model is currently loaded
        return self._loaded is not None

    ### Method : meta()
    @property
    def meta(self) -> Optional[ModelMeta]:
        """
        Returns metadata associated with the loaded model.

        :param:
            None

        :return:
            Optional[ModelMeta]: model metadata if loaded
        """
        return self._loaded.meta if self._loaded else None

    ### Method : set_model()
    def set_model(self, loaded_model: LoadedModel) -> None:
        """
        Assigns a loaded mlflow model to the Predictor instance.

        :param:
            loaded_model LoadedModel: model bundle containing model object and metadata

        :return:
            None
        """
        ### Assign loaded mlflow model
        self._loaded = loaded_model

    ### Method : predict()
    def predict(self, req: PredictRequest) -> PredictOutput:
        """
        Performs inference using the loaded model.

        :param:
            req PredictRequest: validated input request

        :return:
            PredictOutput: predicted label and optional probability
        """
        ### Ensure model has been loaded before inference
        if not self._loaded:
            raise ModelNotLoadedError("Model is not loaded")

        ### Convert request into single-row DataFrame
        df = pd.DataFrame([req.model_dump()])

        model = self._loaded.model

        ### Generate raw prediction from model
        raw_pred = model.predict(df)
        if isinstance(raw_pred, (pd.Series, pd.DataFrame)):
            raw_pred = raw_pred.values
        pred_value = np.array(raw_pred).reshape(-1)[0]

        ### Normalize prediction output to integer label
        try:
            prediction = int(pred_value)
        except Exception:
            ### fallback for unexpected labels
            prediction = 1 if str(pred_value).lower() in {"1", "true", "yes"} else 0

        ### Attempt probability extraction if supported
        probability: Optional[float] = None
        if hasattr(model, "predict_proba"):
            try:
                raw_proba = model.predict_proba(df)
                raw_proba = np.asarray(raw_proba)
                classes = _infer_classes(model, raw_proba.shape[1])
                probability = _pick_positive_proba(raw_proba[0], classes)
            except Exception:
                logger.info("predict_proba failed; returning probability=None", exc_info=True)
                probability = None

        return PredictOutput(prediction=prediction, probability=probability)

### ------------------------------ Helpers ------------------------------ ###
### Helper : _infer_classes()
def _infer_classes(model: Any, n_cols: int) -> list[Any]:
    """
    Attempts to retrieve the class labels from the model or its pipeline steps.

    :param:
        model Any: loaded model object
        n_cols int: number of probability columns

    :return:
        list[Any]: inferred class labels
    """
    if hasattr(model, "classes_"):
        return list(getattr(model, "classes_"))
    if hasattr(model, "named_steps"):
        for _, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "classes_"):
                return list(getattr(step, "classes_"))
    return list(range(n_cols))

### Helper : _pick_positive_proba()
def _pick_positive_proba(proba_row: np.ndarray, classes: list[Any]) -> float:
    """
    Extracts the probability of the positive class from a probability vector.

    :param:
        proba_row np.ndarray: predicted probabilities for each class
        classes list[Any]: class labels corresponding to probability columns

    :return:
        float: probability associated with positive class
    """
    if len(proba_row) == 0:
        return float("nan")

    try:
        if 1 in classes:
            idx = classes.index(1)
            return float(proba_row[idx])
    except Exception:
        pass

    ### Fallback takes max confidence
    return float(np.max(proba_row))
