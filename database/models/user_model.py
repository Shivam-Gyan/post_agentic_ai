from beanie import Document
from pydantic import EmailStr
from datetime import datetime
from typing import Optional

class User(Document):
    name: str
    email: EmailStr
    password_hash: str
    profile_picture: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    integration_token: Optional[str] = None
    is_active: bool = True

    class Settings:
        name = "users"  # collection name