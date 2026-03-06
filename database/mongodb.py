import os

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from database.models.user_model import User
from database.models.conversation_model import Conversation

# MONGO_URI = "mongodb://localhost:27017"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = AsyncIOMotorClient(MONGO_URI)
db = client["blog_agentic_ai"]

async def init_db():
    await init_beanie(
        database=db,  # pyright: ignore[reportArgumentType]
        document_models=[User, Conversation]
    )