# models_ml.py
import uuid
from sqlalchemy import (
    Column, Integer, Text, ForeignKey, TIMESTAMP, func,
    UniqueConstraint, Index, Boolean
)
from sqlalchemy.dialects.postgresql import UUID, DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import text

BaseML = declarative_base()

def ensure_ml_schema(engine):
    with engine.connect() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "ml"'))
        conn.commit()

class FormDB(BaseML):
    __tablename__ = "form"
    __table_args__ = {'schema': 'ml'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    price_level = Column(Integer)
    city = Column(Text)
    open = Column(Boolean, nullable=True)          
    options = Column(JSONB)             
    description = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    predictions = relationship("Prediction", back_populates="form")

class Prediction(BaseML):
    __tablename__ = "prediction"
    __table_args__ = {'schema': 'ml'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    form_id = Column(UUID(as_uuid=True), ForeignKey("ml.form.id", ondelete="CASCADE"), nullable=False)
    k = Column(Integer, nullable=False)
    model_version = Column(Text, nullable=False)
    latency_ms = Column(Integer)
    status = Column(Text, default="ok")

    form = relationship("FormDB", back_populates="predictions")
    items = relationship("PredictionItem", back_populates="prediction", cascade="all, delete-orphan")

class PredictionItem(BaseML):                         
    __tablename__ = "prediction_item"
    __table_args__ = (
        UniqueConstraint("prediction_id", "rank", name="uq_prediction_item_rank"),
        Index("ix_prediction_item_prediction", "prediction_id"),
        Index("ix_prediction_item_etab", "etab_id"),
        {'schema': 'ml'},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("ml.prediction.id", ondelete="CASCADE"), nullable=False)

    rank = Column(Integer, nullable=False)           
    etab_id = Column(Integer, ForeignKey("etab.id_etab"), nullable=False)  
    score = Column(DOUBLE_PRECISION, nullable=False)

    prediction = relationship("Prediction", back_populates="items")
