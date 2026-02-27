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
    Defines the input schema for prediction requests sent to the API.

    :param:
        quantity int: number of items sold
        unit_price float: price per unit
        discount float: discount rate applied to the sale in range [0, 1]
        country str: country name or code
        product str: product name

    :returns:
        PredictRequest: a validated request object containing model input features
    """
    quantity: int = Field(..., ge=1, le=1_000_000, description="Number of items sold")
    unit_price: float = Field(..., ge=0.0, le=1_000_000.0, description="Unit price")
    discount: float = Field(0.0, ge=0.0, le=1.0, description="Discount rate in [0,1]")
    country: str = Field(..., min_length=2, max_length=56, description="Country name/code")
    product: str = Field(..., min_length=1, max_length=100, description="Product name")

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
