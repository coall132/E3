from sqlalchemy.orm import Session, joinedload, selectinload
import models
import traceback
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import os
from fastapi import FastAPI, Depends, HTTPException, Security, status, Query

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)  # pour /auth/token uniquement
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

API_STATIC_KEY = os.getenv("API_STATIC_KEY", "coall")  # pour échanger contre un token
JWT_SECRET = os.getenv("JWT_SECRET", "coall")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))

def create_access_token(subject: str, expires_delta: Optional[timedelta] = None):
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"sub": subject, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt, int(expire.timestamp())

async def get_current_subject(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        sub: Optional[str] = payload.get("sub")
        if sub is None:
            raise credentials_exc
        return sub
    except JWTError:
        raise credentials_exc