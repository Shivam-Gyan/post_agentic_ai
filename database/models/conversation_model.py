from beanie import Document
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class ResponseVersion(BaseModel):
    content: str
    final_blog: Optional[str] = None
    final_checkpoint_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Message(BaseModel):
    role: RoleEnum
    content: str          # For assistant: always latest version. For user: the query.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- USER only ---
    edit_id: Optional[str] = None

    # --- ASSISTANT only ---
    retry_id: Optional[str] = None # same for each turn of conversation (turn is a request <-> response cycle between user and agent)
    final_checkpoint_id: Optional[str] = None  # Latest version's checkpoint — for next turn's parent_checkpoint_id
    final_blog: Optional[str] = None           # Latest version's blog
    versions: list[ResponseVersion] = Field(default_factory=list)  # ← fix


class Conversation(Document):
    thread_id: str               # links to LangGraph checkpoint
    user_id: str                 # links to User
    title: str = "New Chat"
    messages: list[Message] = Field(default_factory=list)
    user_prompts: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    class Settings:
        name = "conversations"
        indexes = [
            "thread_id",         # fast lookup by thread
            "user_id",           # fast lookup by user
        ]