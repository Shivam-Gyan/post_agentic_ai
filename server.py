from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from agent import blog_agentic_ai
from typing import Optional
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr
# from datetime import datetime
import json
from controller.auth_controller import register_user, verify_user
from database.mongodb import init_db
# from database.models.conversation_model import Conversation, Message, RoleEnum
# from database.models.user_model import User
from controller.user_controller import get_user_details
from controller.conversation_controller import *
from middleware.auth_middleware import AuthMiddleware

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
# authenticate the request and verify the jwt token
app.add_middleware(AuthMiddleware)


class GenerateRequest(BaseModel):
    user_query: Optional[str] = None
    mode: Optional[str] = "chat"
    # thread_id: Optional[str] = "blog_generation_thread"
    # user_id: str = "anonymous"


@app.get("/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────── Blog generation endpoint ────────────────────────────

@app.post("/generate/{thread_id}")
async def generate(request: Request):
    """Start a blog generation/refinement/chat turn and stream tokens back.

    Streams SSE (Server-Sent Events) with JSON payloads:
      - {"type": "token", "content": "..."} for each LLM token
      - {"type": "result", ...} for the final state summary
      - {"type": "error", "detail": "..."} on failure
    """

    body_json = await request.json()
    thread_id = request.path_params.get("thread_id")
    # user_id = request.state.user.get("sub")
    user_id = "69ad014ef26c88290f793353"

    try:
        body = GenerateRequest(**(body_json or {}))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request body")
    allowed_modes = {"chat", "generate", "refine", "publish"}

    body.mode = (body.mode or "").strip().lower()
    body.user_query = (body.user_query or "").strip()

    if body.mode not in allowed_modes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid mode. Allowed modes: {', '.join(allowed_modes)}")

    if not body.mode and not body.user_query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide `mode` or `user_query`")

    config = {
        "configurable": {
            "thread_id": thread_id or "blog_generation_thread",
        }
    }

    # Shared mutable container so the generator can pass data out
    stream_result: dict = {}

    async def event_stream():
        try:
            final_state = None
            async for event in blog_agentic_ai.astream_events(
                {"user_query":body.user_query, "mode": body.mode}, config=config, version="v2" #type: ignore[call-arg]
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
                # "title": ,
                "response": assistant_response,
                "final_blog": final_state.get("final_blog") if final_state else None,
            }
            yield f"data: {json.dumps(result)}\n\n"

            # Save conversation to DB after streaming is done
            response = await save_conversation_func(
                user_id=user_id,
                thread_id=thread_id or "blog_generation_thread",
                user_query=body.user_query,  # type: ignore[arg-type]
                assistant_response=assistant_response,
            )

            if not response.get("success"):
                # If DB save failed, send an error event
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Failed to save conversation'})}\n\n"


        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ──────────────────────────── Authentication endpoints ────────────────────────────

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    profile_picture: Optional[str] = None


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(req: CreateUserRequest):
    """this route handle the user register/signup for new user"""
    try:
        response = await register_user(req)
        return response
    except Exception as e:
        return {"message": str(e),"success": False}

class VerifyUserRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/auth/verify",status_code=status.HTTP_200_OK)
async def verify(req: VerifyUserRequest):
    """this route handle the user verification"""
    try:
        response = await verify_user(req)
        return response
    except Exception as e:
        return {"message": str(e),"success": False}

# ──────────────────────── User endpoint ────────────────────────

# get user details by user_id
@app.get("/user/detail")
async def user_details(req: Request):
    """this route handle the user details fetching"""
    try:
        response = await get_user_details(req)
        return response
    except Exception as e:
        return {"message": str(e),"success": False}




# ──────────────────────── Conversation endpoints ────────────────────────

# get all the conversation from the user_id (from Jwt)
@app.get("/conversations")
async def get_user_conversations(req:Request):
    """Return all conversations for a user (metadata only — no messages)."""
    try:
        response = await get_all_conversations_func(req)
        return response
    except Exception as e:
        return {"message": str(e),"success": False}
    
    
# get conversation by thread_id and user_id (from Jwt)
@app.get("/conversations/{thread_id}")
async def get_conversation(req: Request):
    """Return a single conversation with full messages."""
    try:
        response = await get_conversation_by_thread_id_func(req)
        return response
    except Exception as e:
        return {"message": str(e),"success": False}
    
    
# soft delete a conversation by setting is_active to False
@app.post("/conversations/{thread_id}/delete")
async def delete_conversation(req: Request):
    """Soft delete a conversation by setting is_active to False."""
    try:
        response = await delete_conversation_func(req)
        return response
    
    except Exception as e:
        return {"message": str(e),"success": False}

# create a new conversation 

