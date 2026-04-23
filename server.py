
# from contextlib import asynccontextmanager
# import asyncio
# import os
# import logging
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# import agent
# from typing import Optional
# from fastapi import HTTPException, status
# from pydantic import BaseModel, EmailStr
# import json
# from controller.auth_controller import register_user, verify_user
# from database.mongodb import init_db
# from controller.user_controller import get_user_details
# from controller.conversation_controller import *
# from middleware.auth_middleware import AuthMiddleware
# from models import text_to_speech_model
# from utils import strip_markdown, truncate_to_limit
# from dotenv import load_dotenv
# from controller.generate_retry_stream import _run_agent_stream, _run_agent_stream_retry, retry_func, _run_agent_stream_edit
# from uuid import uuid4

# # load_dotenv FIRST — so LOG_LEVEL and other env vars are available immediately
# load_dotenv()

# logging.basicConfig(
#     level=os.getenv("LOG_LEVEL", "INFO").upper(),
#     format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# # Silence noisy third-party libraries
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("groq").setLevel(logging.WARNING)

# logger = logging.getLogger(__name__)


# # ──────────────────────────────────────────────────────────────────────────────
# # Lifespan
# # ──────────────────────────────────────────────────────────────────────────────

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await init_db()
#     await agent.init_blog_graph()
#     yield


# # ──────────────────────────────────────────────────────────────────────────────
# # App + Middleware
# # ──────────────────────────────────────────────────────────────────────────────

# app = FastAPI(title="Blog Agentic AI", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.add_middleware(AuthMiddleware)


# # ──────────────────────────────────────────────────────────────────────────────
# # Health
# # ──────────────────────────────────────────────────────────────────────────────

# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# # ──────────────────────────────────────────────────────────────────────────────
# # Blog generation — SSE streaming endpoint
# # ──────────────────────────────────────────────────────────────────────────────

# ALLOWED_MODES = frozenset({"chat", "generate", "refine", "publish"})

# class GenerateRequest(BaseModel):
#     user_query: str
#     mode: str = "chat"
#     previous_final_checkpoint_id: Optional[str] = None


# @app.post("/generate/{thread_id}")
# async def generate(request: Request):
#     body_json = await request.json()
#     thread_id = request.path_params.get("thread_id")
#     user_id = request.state.user.get("sub")
    
#     body = GenerateRequest(**(body_json or {}))
    
#     return StreamingResponse(
#         _run_agent_stream(
#             user_id=user_id,
#             thread_id=( thread_id if thread_id else str(uuid4())).strip(),
#             mode=body.mode.strip().lower(),
#             user_query=body.user_query.strip(),
#             previous_final_checkpoint_id=body.previous_final_checkpoint_id.strip() if body.previous_final_checkpoint_id else None,
#         ),
#         media_type="text/event-stream"
#     )

# class RetryRequest(BaseModel):
#     user_query: str
#     mode: str = "chat"


# @app.post("/retry/{thread_id}/{checkpoint_id}")
# async def retry_from_checkpoint(request: Request):

#     body_json = await request.json()
#     thread_id = request.path_params.get("thread_id")
#     checkpoint_id = request.path_params.get("checkpoint_id")
#     user_id = request.state.user.get("sub")
    
#     body = RetryRequest(**(body_json or {}))
    
#     return StreamingResponse(
#         _run_agent_stream_retry(
#             user_id=user_id,
#             mode=body.mode.strip().lower(),
#             thread_id=str(thread_id).strip(),
#             user_query=body.user_query.strip(),
#             retry_checkpoint_id= str(checkpoint_id).strip(),
#         ),
#         media_type="text/event-stream"
#     )


# class EditRequest(BaseModel):
#     mode: str
#     new_user_query: str
#     edit_checkpoint_id: str


# @app.post("/edit/{thread_id}/{edit_checkpoint_id}")
# async def edit_from_checkpoint(request: Request):
#     body_json = await request.json()
#     thread_id = request.path_params.get("thread_id")
#     user_id = request.state.user.get("sub")

#     body = EditRequest(**(body_json or {}))

#     return StreamingResponse(
#         _run_agent_stream_edit(
#             user_id=user_id,
#             thread_id=str(thread_id).strip(),
#             mode=body.mode.strip().lower(),
#             edit_checkpoint_id=body.edit_checkpoint_id.strip(),
#             new_user_query=body.new_user_query.strip(),
#         ),
#         media_type="text/event-stream"
#     )

# @app.get("/history/{thread_id}")
# async def get_history(thread_id: str):
#     print(f"Fetching history for thread_id: {thread_id}")
#     return await retry_func(thread_id=thread_id)


# # ──────────────────────────────────────────────────────────────────────────────
# # Auth
# # ──────────────────────────────────────────────────────────────────────────────

# class CreateUserRequest(BaseModel):
#     name: str
#     email: EmailStr
#     password: str
#     profile_picture: Optional[str] = None


# @app.post("/auth/register", status_code=status.HTTP_201_CREATED)
# async def register(req: CreateUserRequest):
#     try:
#         response = await register_user(req)
#         return {
#             "user":      response,
#             "jwt_token": response["jwt_token"],
#             "success":   True,
#             "message":   "User registered successfully",
#         }
#     except Exception as exc:
#         logger.exception("register failed")
#         return {"message": str(exc), "success": False}


# class VerifyUserRequest(BaseModel):
#     email: EmailStr
#     password: str


# @app.post("/auth/verify", status_code=status.HTTP_200_OK)
# async def verify(req: VerifyUserRequest):
#     try:
#         response = await verify_user(req)
#         return {
#             "user":      response,
#             "jwt_token": response["jwt_token"],
#             "success":   True,
#             "message":   "User verified successfully",
#         }
#     except Exception as exc:
#         logger.exception("verify failed")
#         return {"message": str(exc), "success": False}


# # ──────────────────────────────────────────────────────────────────────────────
# # User
# # ──────────────────────────────────────────────────────────────────────────────

# @app.get("/user/detail")
# async def user_details(req: Request):
#     try:
#         return await get_user_details(req)
#     except Exception as exc:
#         logger.exception("user_details failed")
#         return {"message": str(exc), "success": False}


# # ──────────────────────────────────────────────────────────────────────────────
# # Conversations
# # ──────────────────────────────────────────────────────────────────────────────

# @app.get("/conversations")
# async def get_user_conversations(req: Request):
#     try:
#         return await get_all_conversations_func(req)
#     except Exception as exc:
#         logger.exception("get_all_conversations failed")
#         return {"message": str(exc), "success": False}


# @app.get("/conversations/{thread_id}")
# async def get_conversation(req: Request):
#     try:
#         return await get_conversation_by_thread_id_func(req)
#     except Exception as exc:
#         logger.exception("get_conversation failed")
#         return {"message": str(exc), "success": False}


# @app.post("/conversations/{thread_id}/delete")
# async def delete_conversation(req: Request):
#     try:
#         return await delete_conversation_func(req)
#     except Exception as exc:
#         logger.exception("delete_conversation failed")
#         return {"message": str(exc), "success": False}


# @app.delete("/conversations/soft-delete/{thread_id}")
# async def soft_delete_conversation(req: Request):
#     try:
#         return await soft_delete_conversation_func(req)
#     except Exception as exc:
#         logger.exception("soft_delete_conversation failed")
#         return {"message": str(exc), "success": False}


# @app.delete("/conversations/hard-delete/{thread_id}")
# async def hard_delete_conversation(req: Request):
#     try:
#         return await hard_delete_conversation_func(req)
#     except Exception as exc:
#         logger.exception("hard_delete_conversation failed")
#         return {"message": str(exc), "success": False}
    
# # ──────────────────────────────────────────────────────────────────────────────
# # Text to Speech endpoint (optional, can be implemented later)
# # ──────────────────────────────────────────────────────────────────────────────

# @app.post("/tts/{thread_id}")
# async def text_to_speech(request: Request):
#     body_json = await request.json()
#     raw_text: str = body_json.get("text", "").strip()
#     voice: str    = body_json.get("voice", "troy")

#     if not raw_text:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No text provided")

#     # ── Strip markdown before sending to Groq ──
#     clean_text = strip_markdown(raw_text)

#     if not clean_text:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No speakable text after stripping markdown")

#     clean_text = truncate_to_limit(clean_text)

#     logger.info("tts: original_len=%d clean_len=%d", len(raw_text), len(clean_text))
    
#     def audio_stream():
#         with text_to_speech_model.audio.speech.with_streaming_response.create(
#             model="canopylabs/orpheus-v1-english",
#             voice=voice,
#             input=clean_text,   # ← clean text, not raw markdown
#             response_format="wav",
#         ) as response:
#             for chunk in response.iter_bytes(chunk_size=4096):
#                 yield chunk

#     return StreamingResponse(
#         audio_stream(),
#         media_type="audio/wav",
#         headers={"Cache-Control": "no-cache"},
#     )




# main.py
#
# KEY CHANGE vs your original:
#
# Lifespan now calls agent.shutdown_mcp() on exit.
# This cleanly terminates the stdio subprocess that was started in
# init_mcp_tools(). Without this, the uv/fastmcp process becomes a
# zombie when uvicorn restarts or the app is stopped.
# init_mcp_tools() is called implicitly by init_blog_graph() so you
# don't need to call it separately here.

from contextlib import asynccontextmanager
import asyncio
import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import agent
from typing import Optional
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr
import json
from controller.auth_controller import register_user, verify_user
from database.mongodb import init_db
from controller.user_controller import get_user_details
from controller.conversation_controller import *
from middleware.auth_middleware import AuthMiddleware
from models import text_to_speech_model
from utils import strip_markdown, truncate_to_limit
from dotenv import load_dotenv
from controller.generate_retry_stream import (
    _run_agent_stream,
    _run_agent_stream_retry,
    retry_func,
    _run_agent_stream_edit,
)
from uuid import uuid4

# load_dotenv FIRST — so LOG_LEVEL and other env vars are available immediately
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await agent.init_blog_graph()   # internally calls init_mcp_tools() first,
                                    # then builds the graph with cached tools
    yield
    # await agent.shutdown_mcp()      # cleanly kill the stdio subprocess on exit


# ──────────────────────────────────────────────────────────────────────────────
# App + Middleware
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Blog Agentic AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Blog generation — SSE streaming endpoint
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_MODES = frozenset({"chat", "generate", "refine", "publish"})


class GenerateRequest(BaseModel):
    user_query: str
    mode: str = "chat"
    previous_final_checkpoint_id: Optional[str] = None


@app.post("/generate/{thread_id}")
async def generate(request: Request):
    body_json = await request.json()
    thread_id = request.path_params.get("thread_id")
    user_id = request.state.user.get("sub")

    body = GenerateRequest(**(body_json or {}))

    return StreamingResponse(
        _run_agent_stream(
            user_id=user_id,
            thread_id=(thread_id if thread_id else str(uuid4())).strip(),
            mode=body.mode.strip().lower(),
            user_query=body.user_query.strip(),
            previous_final_checkpoint_id=(
                body.previous_final_checkpoint_id.strip()
                if body.previous_final_checkpoint_id
                else None
            ),
        ),
        media_type="text/event-stream",
    )


class RetryRequest(BaseModel):
    user_query: str
    mode: str = "chat"


@app.post("/retry/{thread_id}/{checkpoint_id}")
async def retry_from_checkpoint(request: Request):
    body_json = await request.json()
    thread_id = request.path_params.get("thread_id")
    checkpoint_id = request.path_params.get("checkpoint_id")
    user_id = request.state.user.get("sub")

    body = RetryRequest(**(body_json or {}))

    return StreamingResponse(
        _run_agent_stream_retry(
            user_id=user_id,
            mode=body.mode.strip().lower(),
            thread_id=str(thread_id).strip(),
            user_query=body.user_query.strip(),
            retry_checkpoint_id=str(checkpoint_id).strip(),
        ),
        media_type="text/event-stream",
    )


class EditRequest(BaseModel):
    mode: str
    new_user_query: str
    edit_checkpoint_id: str


@app.post("/edit/{thread_id}/{edit_checkpoint_id}")
async def edit_from_checkpoint(request: Request):
    body_json = await request.json()
    thread_id = request.path_params.get("thread_id")
    user_id = request.state.user.get("sub")

    body = EditRequest(**(body_json or {}))

    return StreamingResponse(
        _run_agent_stream_edit(
            user_id=user_id,
            thread_id=str(thread_id).strip(),
            mode=body.mode.strip().lower(),
            edit_checkpoint_id=body.edit_checkpoint_id.strip(),
            new_user_query=body.new_user_query.strip(),
        ),
        media_type="text/event-stream",
    )


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    print(f"Fetching history for thread_id: {thread_id}")
    return await retry_func(thread_id=thread_id)


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    profile_picture: Optional[str] = None


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(req: CreateUserRequest):
    try:
        response = await register_user(req)
        return {
            "user": response,
            "jwt_token": response["jwt_token"],
            "success": True,
            "message": "User registered successfully",
        }
    except Exception as exc:
        logger.exception("register failed")
        return {"message": str(exc), "success": False}


class VerifyUserRequest(BaseModel):
    email: EmailStr
    password: str


@app.post("/auth/verify", status_code=status.HTTP_200_OK)
async def verify(req: VerifyUserRequest):
    try:
        response = await verify_user(req)
        return {
            "user": response,
            "jwt_token": response["jwt_token"],
            "success": True,
            "message": "User verified successfully",
        }
    except Exception as exc:
        logger.exception("verify failed")
        return {"message": str(exc), "success": False}


# ──────────────────────────────────────────────────────────────────────────────
# User
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/user/detail")
async def user_details(req: Request):
    try:
        return await get_user_details(req)
    except Exception as exc:
        logger.exception("user_details failed")
        return {"message": str(exc), "success": False}


# ──────────────────────────────────────────────────────────────────────────────
# Conversations
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/conversations")
async def get_user_conversations(req: Request):
    try:
        return await get_all_conversations_func(req)
    except Exception as exc:
        logger.exception("get_all_conversations failed")
        return {"message": str(exc), "success": False}


@app.get("/conversations/{thread_id}")
async def get_conversation(req: Request):
    try:
        return await get_conversation_by_thread_id_func(req)
    except Exception as exc:
        logger.exception("get_conversation failed")
        return {"message": str(exc), "success": False}


@app.post("/conversations/{thread_id}/delete")
async def delete_conversation(req: Request):
    try:
        return await delete_conversation_func(req)
    except Exception as exc:
        logger.exception("delete_conversation failed")
        return {"message": str(exc), "success": False}


@app.delete("/conversations/soft-delete/{thread_id}")
async def soft_delete_conversation(req: Request):
    try:
        return await soft_delete_conversation_func(req)
    except Exception as exc:
        logger.exception("soft_delete_conversation failed")
        return {"message": str(exc), "success": False}


@app.delete("/conversations/hard-delete/{thread_id}")
async def hard_delete_conversation(req: Request):
    try:
        return await hard_delete_conversation_func(req)
    except Exception as exc:
        logger.exception("hard_delete_conversation failed")
        return {"message": str(exc), "success": False}


# ──────────────────────────────────────────────────────────────────────────────
# Text to Speech
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/tts/{thread_id}")
async def text_to_speech(request: Request):
    body_json = await request.json()
    raw_text: str = body_json.get("text", "").strip()
    voice: str = body_json.get("voice", "troy")

    if not raw_text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No text provided")

    clean_text = strip_markdown(raw_text)

    if not clean_text:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="No speakable text after stripping markdown",
        )

    clean_text = truncate_to_limit(clean_text)

    logger.info("tts: original_len=%d clean_len=%d", len(raw_text), len(clean_text))

    def audio_stream():
        with text_to_speech_model.audio.speech.with_streaming_response.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice,
            input=clean_text,
            response_format="wav",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache"},
    )