
import os
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

# Use a global client to prevent connection bloat
_client = None

def get_checkpointer() -> MongoDBSaver:
    global _client
    mongo_uri = os.getenv("MONGO_URI") # Or import from your config
    if _client is None:
        _client = MongoClient(mongo_uri)
    
    # This checkpointer will now work perfectly with your async graph
    return MongoDBSaver(_client, db_name="blog_agentic_ai")