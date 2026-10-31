"""Request/response contracts for the API (validated by Pydantic).

SEX, AGE and MARRIAGE are deliberately NOT accepted: they are audit-only
attributes, and ``extra="forbid"`` rejects them if a client sends them.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field



def _pay() -> int:
    return Field(
        ...,
        ge=-2,
        le=9,
        description="Repayment status: -1 = paid duly, 1..9 = months of delay. -2 and 0 are not defined in the official description (commonly read as no consumption / revolving credit).",
    )


def _bill() -> float:
    return Field(..., ge=-1_000_000, le=10_000_000, description="Bill statement amount (NT dollar). Can be negative (credit balance).")


def _paid() -> float:
    return Field(..., ge=0, le=10_000_000, description="Amount paid (NT dollar).")


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 50000, "EDUCATION": 2,
                "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000,
                "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
                "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
                "PAY_AMT4": 1500, "PAY_AMT5": 1500, "PAY_AMT6": 1500,
            }
        },
    )

    LIMIT_BAL: float = Field(..., gt=0, le=10_000_000, description="Credit limit (NT dollar).")
    EDUCATION: int = Field(..., ge=1, le=4, description="1 graduate school, 2 university, 3 high school, 4 other.")

    PAY_1: int = _pay()
    PAY_2: int = _pay()
    PAY_3: int = _pay()
    PAY_4: int = _pay()
    PAY_5: int = _pay()
    PAY_6: int = _pay()

    BILL_AMT1: float = _bill()
    BILL_AMT2: float = _bill()
    BILL_AMT3: float = _bill()
    BILL_AMT4: float = _bill()
    BILL_AMT5: float = _bill()
    BILL_AMT6: float = _bill()

    PAY_AMT1: float = _paid()
    PAY_AMT2: float = _paid()
    PAY_AMT3: float = _paid()
    PAY_AMT4: float = _paid()
    PAY_AMT5: float = _paid()
    PAY_AMT6: float = _paid()


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    risk_band: Literal["Below average", "Around average", "Elevated", "Very high"]
    average_default_rate: float = Field(..., ge=0, le=1)
    flagged_for_review: bool
    threshold: float
    model_version: str
    disclaimer: str = (
        "Statistical estimate for analyst decision support only. "
        "It is not a guarantee and must not be the sole basis for a lending decision."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
