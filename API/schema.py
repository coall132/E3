from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: int

class Form(BaseModel):
    price_level:Optional[int]
    city:Optional[str]
    open:Optional[str]
    options:Optional[list]
    description:Optional[str]
    created_at: Optional[datetime] = Field(default=None, json_schema_extra={"readOnly": True})

class PredictionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    id: Optional[UUID] = Field(default=None, json_schema_extra={"readOnly": True})
    prediction_id: Optional[UUID] = Field(default=None, json_schema_extra={"readOnly": True})
    rank: int
    etab_id: int
    score: float

class Prediction(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    id: Optional[UUID] = Field(default=None, json_schema_extra={"readOnly": True})
    form_id: UUID
    k: int = Field(10, ge=1, le=50)
    model_version: Optional[str] = Field(default=None, json_schema_extra={"readOnly": True})
    latency_ms: Optional[int] = Field(default=None, json_schema_extra={"readOnly": True})
    status: Optional[str] = "ok"

    items: List[PredictionItem] = []
