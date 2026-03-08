from datetime import datetime, timedelta
from jose import jwt
import os

SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-this")
ALGORITHM = "HS256"


def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(days=7)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt