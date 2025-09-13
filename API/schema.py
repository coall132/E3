from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from fastapi import HTTPException

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Message d’erreur lisible.")
    code: str | None = Field(None, description="Code d’erreur applicatif (optionnel).")

class TokenOut(BaseModel):
    access_token: str = Field(description="JWT d’accès.")
    token_type: str = "bearer"
    expires_at: int

class ApiKeyCreate(BaseModel):
    email: str = Field( description="Email de l’utilisateur.")
    username: str = Field(min_length=3, max_length=50, description="Nom d’utilisateur (unique).")
    model_config = ConfigDict(json_schema_extra={
        "example": {"email": "alice@example.com", "username": "alice"}
    })

class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    api_key: str = Field(description="Clé API en clair. Stocke-la côté client !")                
    key_id: str = Field(description="Identifiant public de la clé.")             


class Form(BaseModel):
    price_level:Optional[int] = Field( description="Niveau de prix 1–4.")
    city:Optional[str] = Field(None, alias="code_postal",description="Code postal.")
    open:Optional[str] = Field(None, description="Filtrer sur établissements ouverts maintenant.")
    options:Optional[list] = Field(default_factory=list, description="Options (ex: ['delivery','restroom']).")
    description:Optional[str] = Field(None, description="Texte libre décrivant l’envie.")
    created_at: Optional[datetime] = Field(default=None, json_schema_extra={"readOnly": True})
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "price_level": 2,
            "city": "37000",
            "open": True,
            "options": ["delivery", "goodForChildren"],
            "description": "italien cosy avec terrasse, budget modéré"
        }
    })

class Geo(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None

class EtablissementDetails(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id_etab: int
    nom: Optional[str] = None
    adresse: Optional[str] = None
    telephone: Optional[str] = None
    site_web: Optional[str] = None
    description: Optional[str] = None
    rating: Optional[float] = None
    priceLevel: Optional[int] = None
    priceLevel_symbole: Optional[str] = None
    startPrice: Optional[float] = None
    endPrice: Optional[float] = None
    geo: Optional[Geo] = None
    options_actives: List[str] = []
    horaires: List[str] = []

class PredictionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    rank: int
    etab_id: int
    score: float

class PredictionItemRich(PredictionItem):
    details: Optional[EtablissementDetails] = None

class Prediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    form_id: UUID
    k: int
    model_version: Optional[str] = None
    latency_ms: Optional[int] = None
    status: Optional[str] = "ok"
    items: List[PredictionItem] = []

class PredictionResponse(Prediction):
    prediction_id: UUID
    items_rich: List[PredictionItemRich] = []
    message: Optional[str] = None

class FeedbackIn(BaseModel):
    prediction_id: UUID  = Field(description="ID de la prédiction à évaluer.")
    rating: Optional[int] = Field(None,ge=0, le=5,description="ID de la prédiction à évaluer.")     
    comment: Optional[str] = Field(None, description="Commentaire libre.")

class FeedbackOut(BaseModel):
    status: str = "ok"

@dataclass
class MLState:
    preproc: object | None = None          
    preproc_factory: object | None = None 
    sent_model: object | None = None
    rank_model: object | None = None
    rank_model_path: str | None = None