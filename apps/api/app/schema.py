"""
apps/api/app/schema.py

Authors:
    BAUDET Quentin
    CARDONA Quentin
    LARMAILLARD-NOIREN Joris
"""
### Modules importation
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

### ------------------------------ Classes ------------------------------ ###
### Class : PredictRequest
class PredictRequest(BaseModel):
    """
    Defines the validated input schema for prediction requests sent to the API.

    :param:
        sales_person str: name of the salesperson associated with the transaction
        country str: country name or ISO code
        product str: product name
        boxes_shipped int: number of boxes shipped
        date str: transaction date formatted as YYYY-MM-DD

    :returns:
        PredictRequest: a validated request object containing model input features
    """
    sales_person: str = Field(..., min_length=1, max_length=120)
    country: str = Field(..., min_length=2, max_length=56)
    product: str = Field(..., min_length=1, max_length=120)
    boxes_shipped: int = Field(..., ge=0, le=1_000_000)
    date: str = Field(..., description="YYYY-MM-DD")

    model_config = ConfigDict(protected_namespaces=())

### Class : PredictResponse
class PredictResponse(BaseModel):
    """
    Defines the output schema returned by the prediction endpoint.

    :param:
        prediction int: predicted class label
        probability float: confidence score associated with the prediction
        model_stage str: mlflow model stage used for inference
        model_version str: mlflow model version used for inference

    :returns:
        PredictResponse: structured response containing prediction results and model metadata
    """
    prediction: int
    probability: Optional[float] = None

    model_name: str
    model_stage: str
    model_version: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())
