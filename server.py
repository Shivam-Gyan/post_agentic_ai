from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from agent import blog_agentic_ai
from typing import Optional
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr
from datetime import datetime
import json

from database.mongodb import init_db
from database.models.conversation_model import Conversation, Message, RoleEnum
from database.models.user_model import User

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Blog Agentic AI", lifespan=lifespan)

# Allow CORS from any origin (use cautiously in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    user_query: Optional[str] = None
    mode: Optional[str] = "chat"
    thread_id: Optional[str] = "blog_generation_thread"
    user_id: str = "anonymous"

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Start a blog generation/refinement/chat turn and stream tokens back.

    Streams SSE (Server-Sent Events) with JSON payloads:
      - {"type": "token", "content": "..."} for each LLM token
      - {"type": "result", ...} for the final state summary
      - {"type": "error", "detail": "..."} on failure
    """
    if req.mode and req.user_query:
        user_query = f"{req.mode}: {req.user_query}"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide `blog_description` or `user_query`")

    config = {
        "configurable": {
            "thread_id": req.thread_id or "blog_generation_thread",
        }
    }

    # Shared mutable container so the generator can pass data out
    stream_result: dict = {}

    async def event_stream():
        try:
            final_state = None
            async for event in blog_agentic_ai.astream_events(
                {"user_query": user_query}, config=config, version="v2"
            ):
                kind = event["event"]

                # Stream individual LLM tokens
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    content = chunk.content if chunk else None
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                # Capture the final state when the graph ends
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output", {})

            # Extract assistant response
            assistant_response = (
                final_state["messages"][-1].content
                if final_state and final_state.get("messages")
                else None
            )

            # Store for post-stream DB save
            stream_result["assistant_response"] = assistant_response
            stream_result["final_state"] = final_state

            # Send the final result summary
            result = {
                "type": "result",
                "mode": final_state.get("mode") if final_state else None,
                "response": assistant_response,
                "final_blog": final_state.get("final_blog") if final_state else None,
            }
            yield f"data: {json.dumps(result)}\n\n"

            # Save conversation to DB after streaming is done
            await _save_conversation(
                user_id=req.user_id,
                thread_id=req.thread_id or "blog_generation_thread",
                user_query=req.user_query,  # type: ignore[arg-type]
                assistant_response=assistant_response,
            )

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _save_conversation(
    user_id: str, thread_id: str, user_query: str, assistant_response: str | None
):
    """Persist the user prompt and assistant reply into the conversations collection."""
    now = datetime.utcnow()

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


# ──────────────────────────── User endpoints ────────────────────────────

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    profile_picture: Optional[str] = None


@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(req: CreateUserRequest):
    existing = await User.find_one(User.email == req.email)
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    user = User(name=req.name, email=req.email, profile_picture=req.profile_picture)
    await user.insert()
    return {"id": str(user.id), "name": user.name, "email": user.email}


# ──────────────────────── Conversation endpoints ────────────────────────

@app.get("/users/{user_id}/conversations")
async def get_user_conversations(user_id: str):
    """Return all conversations for a user (metadata only — no messages)."""
    conversations = await Conversation.find(
        Conversation.user_id == user_id
    ).sort("-updated_at").to_list()

    return [
        {
            "id": str(c.id),
            "thread_id": c.thread_id,
            "title": c.title,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
            "is_active": c.is_active,
            "message_count": len(c.messages),
        }
        for c in conversations
    ]


@app.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """Return a single conversation with full messages."""
    conversation = await Conversation.find_one(Conversation.thread_id == thread_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": str(conversation.id),
        "thread_id": conversation.thread_id,
        "user_id": conversation.user_id,
        "title": conversation.title,
        "messages": [m.model_dump() for m in conversation.messages],
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }