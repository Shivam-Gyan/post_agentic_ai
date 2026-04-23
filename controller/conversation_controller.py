


from datetime import datetime, timezone

from fastapi import HTTPException,Request
from database.models.conversation_model import Conversation, Message, ResponseVersion, RoleEnum
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

# async def save_conversation_func(
#     user_id: str,
#     thread_id: str,
#     user_query: str,
#     assistant_response: str | None,
#     assistant_response_blog: str | None,
#     edit_checkpoint_id: str | None,
#     retry_checkpoint_id: str | None,
#     final_checkpoint_id: str | None,
#     is_retry: bool = False,
#     **kwargs
# ):
#     """Persist the user prompt and assistant reply into the conversations collection."""
#     now = datetime.now(timezone.utc)

#     try:
#         new_messages = []

#         if user_query:
#             new_messages.append(Message(
#                 role=RoleEnum.user,
#                 content=user_query,
#                 timestamp=now,
#                 checkpoint_id=edit_checkpoint_id,
#             ))

#         if assistant_response:
#             new_messages.append(Message(
#                 role=RoleEnum.assistant,
#                 content=assistant_response,
#                 final_blog=assistant_response_blog,
#                 checkpoint_id=retry_checkpoint_id,
#                 final_checkpoint_id=final_checkpoint_id,
#                 timestamp=now,
#             ))

#         if not new_messages:
#             logger.warning("save_conversation_func: nothing to save for thread=%s", thread_id)
#             return {"success": True}

#         conversation = await Conversation.find_one(
#             Conversation.thread_id == thread_id,
#             Conversation.user_id == user_id,
#         )

#         if conversation:
#             conversation.messages.extend(new_messages)
#             if user_query:
#                 conversation.user_prompts.append(user_query)
#             conversation.updated_at = now
#             await conversation.save()
#         else:
#             conversation = Conversation(
#                 thread_id=thread_id,
#                 user_id=user_id,
#                 title=user_query[:50] if user_query else "New Chat",
#                 messages=new_messages,
#                 user_prompts=[user_query] if user_query else [],
#                 created_at=now,
#                 updated_at=now
#             )
#             await conversation.insert()

#         return {"success": True}

#     except Exception as e:
#         logger.exception("save_conversation_func failed for thread=%s", thread_id)
#         return {"success": False, "message": str(e)}
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
    


# save conversation to db 
async def save_conversation_func(
    user_id: str,
    thread_id: str,
    user_query: str,
    assistant_response: str | None,
    assistant_response_blog: str | None,
    edit_checkpoint_id: str | None,
    retry_checkpoint_id: str | None,
    final_checkpoint_id: str | None,
):
    now = datetime.now(timezone.utc)

    try:
        new_messages = []

        if user_query:
            new_messages.append(Message(
                role=RoleEnum.user,
                content=user_query,
                timestamp=now,
                edit_id=edit_checkpoint_id,      # ← explicit field
            ))

        if assistant_response:
            v0 = ResponseVersion(
                content=assistant_response,
                final_blog=assistant_response_blog,
                final_checkpoint_id=final_checkpoint_id,
                timestamp=now,
            )
            new_messages.append(Message(
                role=RoleEnum.assistant,
                content=assistant_response,
                final_blog=assistant_response_blog,
                retry_id=retry_checkpoint_id,
                final_checkpoint_id=final_checkpoint_id,
                versions=[v0],                  # ← wrap in list
                timestamp=now,
            ))

        if not new_messages:
            logger.warning("save_conversation_func: nothing to save for thread=%s", thread_id)
            return {"success": True}

        conversation = await Conversation.find_one(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user_id,
        )

        if conversation:
            conversation.messages.extend(new_messages)
            if user_query:
                conversation.user_prompts.append(user_query)
            conversation.updated_at = now
            await conversation.save()
        else:
            conversation = Conversation(
                thread_id=thread_id,
                user_id=user_id,
                title=user_query[:50] if user_query else "New Chat",
                messages=new_messages,
                user_prompts=[user_query] if user_query else [],
                created_at=now,
                updated_at=now,
            )
            await conversation.insert()

        return {"success": True}

    except Exception as e:
        logger.exception("save_conversation_func failed for thread=%s", thread_id)
        return {"success": False, "message": str(e)}
    



#  save retry version od assistant response to db
async def save_retry_version(
    user_id: str,
    thread_id: str,
    assistant_response: str | None,
    assistant_response_blog: str | None,
    final_checkpoint_id: str | None,
    retry_checkpoint_id: str | None,  # used for safety check only
):
    now = datetime.now(timezone.utc)

    try:
        conversation = await Conversation.find_one(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user_id,
        )

        if not conversation:
            logger.error("save_retry_version: conversation not found for thread=%s", thread_id)
            return {"success": False, "message": "Conversation not found"}

        # Find the last assistant message — that's always the one being retried
        last_assistant_idx: int | None = None
        for i in range(len(conversation.messages) - 1, -1, -1):
            if conversation.messages[i].role == RoleEnum.assistant:
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            logger.error("save_retry_version: no assistant message found for thread=%s", thread_id)
            return {"success": False, "message": "No assistant message to retry"}

        msg = conversation.messages[last_assistant_idx]

        # Safety check — make sure retry_id matches
        if retry_checkpoint_id and msg.retry_id != retry_checkpoint_id:
            logger.warning(
                "save_retry_version: retry_id mismatch for thread=%s expected=%s got=%s",
                thread_id, msg.retry_id, retry_checkpoint_id
            )
            # Don't hard fail — mismatch is a warning, not a blocker

        # Build new version
        new_version = ResponseVersion(
            content=assistant_response or "",
            final_blog=assistant_response_blog,
            final_checkpoint_id=final_checkpoint_id,
            timestamp=now,
        )

        # Append version
        msg.versions.append(new_version)

        # Update top-level mirrors to reflect latest version
        msg.content = assistant_response or ""
        msg.final_blog = assistant_response_blog
        msg.final_checkpoint_id = final_checkpoint_id  # points to newest version

        conversation.messages[last_assistant_idx] = msg
        conversation.updated_at = now
        await conversation.save()

        return {"success": True, "version_index": len(msg.versions) - 1}

    except Exception as e:
        logger.exception("save_retry_version failed for thread=%s", thread_id)
        return {"success": False, "message": str(e)}
    


# async def save_edit_turn(
#     user_id: str,
#     thread_id: str,
#     new_user_query: str,
#     assistant_response: str | None,
#     assistant_response_blog: str | None,
#     edit_checkpoint_id: str | None,
#     retry_checkpoint_id: str | None,
#     final_checkpoint_id: str | None,
# ):
#     now = datetime.now(timezone.utc)

#     try:
#         conversation = await Conversation.find_one(
#             Conversation.thread_id == thread_id,
#             Conversation.user_id == user_id,
#         )

#         if not conversation:
#             # Fresh conversation — treat like a normal new turn
#             logger.warning("save_edit_turn: no conversation found, creating new for thread=%s", thread_id)
#             return await save_conversation_func(
#                 user_id=user_id,
#                 thread_id=thread_id,
#                 user_query=new_user_query,
#                 assistant_response=assistant_response,
#                 assistant_response_blog=assistant_response_blog,
#                 edit_checkpoint_id=edit_checkpoint_id,
#                 retry_checkpoint_id=retry_checkpoint_id,
#                 final_checkpoint_id=final_checkpoint_id,
#             )

#         # ✅ Find the last user message index — we replace from there
#         last_user_idx: int | None = None
#         for i in range(len(conversation.messages) - 1, -1, -1):
#             if conversation.messages[i].role == RoleEnum.user:
#                 last_user_idx = i
#                 break

#         if last_user_idx is None:
#             # No user message found — just append normally
#             logger.warning("save_edit_turn: no user message found for thread=%s", thread_id)
#             return await save_conversation_func(
#                 user_id=user_id,
#                 thread_id=thread_id,
#                 user_query=new_user_query,
#                 assistant_response=assistant_response,
#                 assistant_response_blog=assistant_response_blog,
#                 edit_checkpoint_id=edit_checkpoint_id,
#                 retry_checkpoint_id=retry_checkpoint_id,
#                 final_checkpoint_id=final_checkpoint_id,
#             )

#         # ✅ Truncate everything from last user message onwards
#         # This removes the old user message AND the old assistant message
#         conversation.messages = conversation.messages[:last_user_idx]

#         # ✅ Build fresh user + assistant messages (same as save_conversation_func)
#         new_messages = []

#         new_messages.append(Message(
#             role=RoleEnum.user,
#             content=new_user_query,
#             timestamp=now,
#             edit_id=edit_checkpoint_id,
#         ))

#         if assistant_response:
#             v0 = ResponseVersion(
#                 content=assistant_response,
#                 final_blog=assistant_response_blog,
#                 final_checkpoint_id=final_checkpoint_id,
#                 timestamp=now,
#             )
#             new_messages.append(Message(
#                 role=RoleEnum.assistant,
#                 content=assistant_response,
#                 final_blog=assistant_response_blog,
#                 retry_id=retry_checkpoint_id,
#                 final_checkpoint_id=final_checkpoint_id,
#                 versions=[v0],
#                 timestamp=now,
#             ))

#         conversation.messages.extend(new_messages)

#         # ✅ Also update user_prompts — replace last prompt with new one
#         if conversation.user_prompts:
#             conversation.user_prompts[-1] = new_user_query
#         else:
#             conversation.user_prompts.append(new_user_query)

#         conversation.updated_at = now
#         await conversation.save()

#         return {"success": True}

#     except Exception as e:
#         logger.exception("save_edit_turn failed for thread=%s", thread_id)
#         return {"success": False, "message": str(e)}



async def save_edit_turn(
    user_id: str,
    thread_id: str,
    new_user_query: str,
    assistant_response: str | None,
    assistant_response_blog: str | None,
    edit_checkpoint_id_new: str | None,  # new edit_id for the new user message
    edit_checkpoint_id: str | None,
    retry_checkpoint_id: str | None,
    final_checkpoint_id: str | None,
):
    now = datetime.now(timezone.utc)

    try:
        conversation = await Conversation.find_one(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user_id,
        )

        # ─────────────────────────────────────────────
        # CASE 1: No conversation → fallback
        # ─────────────────────────────────────────────
        if not conversation:
            logger.warning("No conversation found, creating new.")
            return await save_conversation_func(
                user_id=user_id,
                thread_id=thread_id,
                user_query=new_user_query,
                assistant_response=assistant_response,
                assistant_response_blog=assistant_response_blog,
                edit_checkpoint_id=edit_checkpoint_id_new,
                retry_checkpoint_id=retry_checkpoint_id,
                final_checkpoint_id=final_checkpoint_id,
            )

        # ─────────────────────────────────────────────
        # STEP 1: Find EXACT user message using edit_id
        # ─────────────────────────────────────────────
        target_user_idx = None

        for i, msg in enumerate(conversation.messages):
            if (
                msg.role == RoleEnum.user
                and msg.edit_id == edit_checkpoint_id
            ):
                target_user_idx = i
                break

        # ─────────────────────────────────────────────
        # STEP 2: Fallback if not found
        # ─────────────────────────────────────────────
        if target_user_idx is None:
            logger.warning("Edit target not found → fallback to last user message")

            for i in range(len(conversation.messages) - 1, -1, -1):
                if conversation.messages[i].role == RoleEnum.user:
                    target_user_idx = i
                    break

        if target_user_idx is None:
            logger.error("No user message found at all")
            return {"success": False, "message": "No user message found"}

        # ─────────────────────────────────────────────
        # STEP 3: TRUNCATE conversation properly
        # ─────────────────────────────────────────────
        conversation.messages = conversation.messages[:target_user_idx]

        # ─────────────────────────────────────────────
        # STEP 4: Build new messages
        # ─────────────────────────────────────────────
        new_messages = []

        new_messages.append(Message(
            role=RoleEnum.user,
            content=new_user_query,
            timestamp=now,
            edit_id=edit_checkpoint_id_new,
        ))

        if assistant_response:
            v0 = ResponseVersion(
                content=assistant_response,
                final_blog=assistant_response_blog,
                final_checkpoint_id=final_checkpoint_id,
                timestamp=now,
            )

            new_messages.append(Message(
                role=RoleEnum.assistant,
                content=assistant_response,
                final_blog=assistant_response_blog,
                retry_id=retry_checkpoint_id,
                final_checkpoint_id=final_checkpoint_id,
                versions=[v0],
                timestamp=now,
            ))

        conversation.messages.extend(new_messages)

        # ─────────────────────────────────────────────
        # STEP 5: FIX user_prompts (CRITICAL)
        # ─────────────────────────────────────────────
        user_prompt_index = 0

        for i, msg in enumerate(conversation.messages):
            if msg.role == RoleEnum.user:
                if i == target_user_idx:
                    break
                user_prompt_index += 1

        conversation.user_prompts = conversation.user_prompts[:user_prompt_index]
        conversation.user_prompts.append(new_user_query)

        # ─────────────────────────────────────────────
        conversation.updated_at = now
        await conversation.save()

        return {"success": True}

    except Exception as e:
        logger.exception("save_edit_turn failed")
        return {"success": False, "message": str(e)}