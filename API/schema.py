from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: int

class ApiKeyCreate(BaseModel):
    email: str
    username: str = Field(min_length=3, max_length=50)

class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    api_key: str                 
    key_id: str              


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

class FeedbackIn(BaseModel):
    prediction_id: UUID
    etab_id: Optional[int] = None    
    rating: Optional[int] = None     
    comment: Optional[str] = None

class FeedbackOut(BaseModel):
    status: str = "ok"

@dataclass
class MLState:
    preproc: object | None = None          
    preproc_factory: object | None = None 
    sent_model: object | None = None
    rank_model: object | None = None
    rank_model_path: str | None = None