# models_ml.py
import uuid
from sqlalchemy import (
    Column, Integer, Text, ForeignKey, TIMESTAMP, func,
    UniqueConstraint, Index, Boolean,String, JSON, DateTime
)
from sqlalchemy.dialects.postgresql import UUID, DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import text
from sqlalchemy import Column, Integer, TIMESTAMP, func, ForeignKey

Base = declarative_base()

def ensure_ml_schema(engine):
    with engine.connect() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "ml"'))
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "user_base"'))
        conn.commit()

class FormDB(Base):
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

class Prediction(Base):
    __tablename__ = "prediction"
    __table_args__ = {'schema': 'ml'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    form_id = Column(UUID(as_uuid=True), ForeignKey("ml.form.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("user_base.user.id", ondelete="RESTRICT"), nullable=False)
    k = Column(Integer, nullable=False)
    model_version = Column(Text, nullable=False)
    latency_ms = Column(Integer)
    status = Column(Text, default="ok")

    form = relationship("FormDB", back_populates="predictions")
    items = relationship("PredictionItem", back_populates="prediction", cascade="all, delete-orphan")

class PredictionItem(Base):                         
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

class User(Base):
    __tablename__ = "user"
    __table_args__ = {'schema': 'user_base'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(80), unique=True, index=True, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")

class ApiKey(Base):
    __tablename__ = "api_key"
    __table_args__ = (
        UniqueConstraint("key_id", name="uq_api_key_keyid"),
        Index("ix_api_key_user", "user_id"),{'schema': 'user_base'}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("user_base.user.id", ondelete="CASCADE"), nullable=False)
    key_id = Column(String(32), nullable=False, index=True)
    key_hash = Column(Text, nullable=False)       
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(TIMESTAMP(timezone=True), nullable=True)

    user = relationship("User", back_populates="api_keys")

class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = {'schema': 'ml'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prediction_id = Column(UUID(as_uuid=True),ForeignKey("ml.prediction.id", ondelete="CASCADE"),
                           nullable=False, index=True)
 
    rating = Column(Integer, nullable=True)           
    comment = Column(Text, nullable=True)
