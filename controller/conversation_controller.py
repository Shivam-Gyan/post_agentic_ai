


from datetime import datetime

from fastapi import HTTPException,Request
from database.models.conversation_model import Conversation, Message, RoleEnum, RoleEnum
from pymongo.errors import PyMongoError
import logging
logger = logging.getLogger(__name__)

# get all the conversation from the user_id (from Jwt)
async def get_all_conversations_func(req:Request):
    """Return all conversations for a user (metadata only — no messages)."""

    try:
        user_id:str = req.state.user.get("sub")

        conversations = await Conversation.find(
            Conversation.user_id == user_id,
            Conversation.is_active == True
        ).sort("-updated_at").to_list()

        return {
            "conversations":[
                {   
                    "thread_id": c.thread_id,
                    "title": c.title,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                    "is_active": c.is_active,
                    "message_count": len(c.messages),
                }
                for c in conversations
            ],
            "success": True
        }

    except PyMongoError as e:
        logger.exception("DB error during register")
        raise HTTPException(status_code=503, detail="Database unavailable")


# get conversation by thread_id and user_id (from Jwt)
async def get_conversation_by_thread_id_func(req:Request):
    """Return a single conversation with full messages."""

    try:
        user_id:str = req.state.user.get("sub")
        thread_id = req.path_params.get("thread_id")

        conversation = await Conversation.find_one(Conversation.thread_id == thread_id, Conversation.user_id == user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "conversation":{
                "thread_id": conversation.thread_id,
                "title": conversation.title,
                "messages": [m.model_dump() for m in conversation.messages],
                "user_prompts": conversation.user_prompts,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
            },
            "success": True
        }
    
    except PyMongoError as e:
        logger.exception("DB error during register")
        raise HTTPException(status_code=503, detail="Database unavailable")



# soft delete a conversation by setting is_active to False
async def delete_conversation_func(req:Request):
    """Soft delete a conversation by setting is_active to False."""

    try:
        user_id:str = req.state.user.get("sub")
        thread_id = req.path_params.get("thread_id")

        conversation = await Conversation.find_one(Conversation.thread_id == thread_id, Conversation.user_id == user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.is_active = False
        conversation.updated_at = datetime.utcnow()
        await conversation.save()

        return {"message": "Conversation deleted successfully", "success": True}
    
    except PyMongoError as e:
        logger.exception("DB error during register")
        raise HTTPException(status_code=503, detail="Database unavailable")
    


# create a new conversation (while generation of response by AI stream)

async def save_conversation_func(
    user_id: str, thread_id: str, user_query: str, assistant_response: str | None
):
    """Persist the user prompt and assistant reply into the conversations collection."""
    now = datetime.utcnow()

    try:
        conversation = await Conversation.find_one(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user_id,
        )

        user_msg = Message(role=RoleEnum.user, content=user_query, timestamp=now)
        new_messages = [user_msg]

        if assistant_response:
            assistant_msg = Message(role=RoleEnum.assistant, content=assistant_response, timestamp=now)
            new_messages.append(assistant_msg)

        if conversation:
            conversation.messages.extend(new_messages)
            conversation.user_prompts.append(user_query)
            conversation.updated_at = now
            await conversation.save()
        else:
            conversation = Conversation(
                thread_id=thread_id,
                user_id=user_id,
                title=user_query[:50] if user_query else "New Chat",
                messages=new_messages,
                user_prompts=[user_query],
                created_at=now,
                updated_at=now,
            )
            await conversation.insert()

        return {"success": True}
    
    except PyMongoError as e:
        logger.exception("DB error during register")
        return {"success": False, "message": "Database error occurred while saving conversation"}

# soft delete a conversation by setting is_active to False
async def soft_delete_conversation_func(req:Request):
    """Soft delete a conversation by setting is_active to False."""

    try:
        user_id:str = req.state.user.get("sub")
        thread_id = req.path_params.get("thread_id")

        conversation = await Conversation.find_one(Conversation.thread_id == thread_id, Conversation.user_id == user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.is_active = False
        conversation.updated_at = datetime.utcnow()
        await conversation.save()

        return {"message": "Conversation soft deleted successfully", "success": True}
    
    except PyMongoError as e:
        logger.exception("DB error during register")
        raise HTTPException(status_code=503, detail="Database unavailable")
    
# hard delete a conversation by removing it from DB
async def hard_delete_conversation_func(req:Request):
    """Hard delete a conversation by removing it from DB."""

    try:
        user_id:str = req.state.user.get("sub")
        thread_id = req.path_params.get("thread_id")

        conversation = await Conversation.find_one(Conversation.thread_id == thread_id, Conversation.user_id == user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        await conversation.delete()

        return {"message": "Conversation hard deleted successfully", "success": True}

    except PyMongoError as e:
        logger.exception("DB error during register")
        raise HTTPException(status_code=503, detail="Database unavailable")
    