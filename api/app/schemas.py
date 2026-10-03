from enum import Enum
from pydantic import BaseModel, Field

class LabelEnum(str, Enum):
    APP_LOGIN = "APP_LOGIN"
    APP_TECH = "APP_TECH"
    CARD_ISSUE = "CARD_ISSUE"
    PAYMENT_OUT_FAIL = "PAYMENT_OUT_FAIL"
    INCOMING_DELAY = "INCOMING_DELAY"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"
    ACCOUNT_SEIZED = "ACCOUNT_SEIZED"
    KYC_VERIFICATION = "KYC_VERIFICATION"

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class ClassifyResponse(BaseModel):
    label: LabelEnum
    score: float = Field(..., ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
