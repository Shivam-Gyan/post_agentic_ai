from beanie import Document
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from enum import Enum

class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class Message(BaseModel):
    role: RoleEnum
    content: str
    timestamp: datetime = datetime.utcnow()

class Conversation(Document):
    thread_id: str               # links to LangGraph checkpoint
    user_id: str                 # links to User
    title: str = "New Chat"
    messages: list[Message] = []
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    is_active: bool = True

    class Settings:
        name = "conversations"
        indexes = [
            "thread_id",         # fast lookup by thread
            "user_id",           # fast lookup by user
        ]