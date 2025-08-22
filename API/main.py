from fastapi import FastAPI, Depends, HTTPException, Security, status, Query
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os

import CRUD
import models
import schema
from database import engine, get_db

app = FastAPI(
    title="API Reco Restaurant",
    description="API pour recommander des restaurants",
    version="1.0.0"
)