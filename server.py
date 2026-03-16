
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

# load_dotenv FIRST — so LOG_LEVEL and other env vars are available immediately
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Silence noisy third-party libraries
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
    await agent.init_blog_graph()
    yield


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
# Node classification sets  (module-level — built once at import time)
# ──────────────────────────────────────────────────────────────────────────────

# Nodes that run LLMs internally but whose output is NEVER shown to the user.
# Tokens from these nodes are classified as "reasoning" in the SSE stream.
REASONING_NODES: frozenset[str] = frozenset({
    "intent_node",
    "router_node",
    "research_node",
    "orchestrator",
    "worker",
    "reducer",
    "refine_structured_output_node",
})

# Nodes whose tokens ARE the final user-facing answer.
ANSWER_NODES: frozenset[str] = frozenset({
    "chat_node",   # inside conversation_subgraph
    "refine_node", # inside refine_subgraph
})

# Verbose UI labels per node — shown as "thinking" steps to the frontend.
# Kept at module level so it is not rebuilt on every on_chain_start event.
VERBOSE_NODE_LABELS: dict[str, str] = {
    "intent_node":                   "Detecting intent...",
    "router_node":                   "Deciding generation strategy...",
    "research_node":                 "Researching topic...",
    "orchestrator":                  "Planning blog sections...",
    "reducer":                       "Combining section drafts...",
    "refine_structured_output_node": "Parsing refinement instructions...",
    "refine_node":                   "Refining blog...",
    # chat_node intentionally omitted — on_chat_model_start sends "Thinking..." instead
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_inner_node_event(event: dict) -> bool:
    """Return True only for the *inner* on_chain_start fired when the node
    function actually executes.

    LangGraph fires on_chain_start TWICE per node:
      1. Outer — pregel scheduler dispatching the node. No checkpoint_ns.
      2. Inner — node function starts running. checkpoint_ns is present.

    Acting on both sends duplicate verbose messages to the client.
    The checkpoint_ns key is the only reliable discriminator between the two.
    """
    return "checkpoint_ns" in event.get("metadata", {})


def sse(payload: dict) -> str:
    """Wrap a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(payload)}\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    user_query: Optional[str] = None
    mode: Optional[str] = "chat"


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


@app.post("/generate/{thread_id}")
async def generate(request: Request):
    """Stream a blog generation / refinement / chat turn as SSE.

    SSE event types:
      {"type": "verbose",   "content": "Detecting intent..."}   — thinking steps
      {"type": "token",     "content": "..."}                   — answer tokens
      {"type": "reasoning", "content": "..."}                   — internal LLM tokens
      {"type": "result",    "response": "...", "final_blog": "..."} — final state
      {"type": "error",     "detail":  "..."}                   — failure
    """
    body_json = await request.json()
    thread_id: str = request.path_params.get("thread_id") # type: ignore[call-arg]
    user_id: str = request.state.user.get("sub")

    try:
        body = GenerateRequest(**(body_json or {}))
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid request body")

    body.mode = (body.mode or "").strip().lower()
    body.user_query = (body.user_query or "").strip()

    if body.mode not in ALLOWED_MODES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode. Allowed: {', '.join(sorted(ALLOWED_MODES))}",
        )
    if not body.mode and not body.user_query:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Provide `mode` or `user_query`",
        )

    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        final_state: dict | None = None

        # Per-request deduplication state
        seen_verbose: set[str] = set()
        worker_count: int = 0

        try:
            async for event in agent.blog_agentic_ai.astream_events(
                {"user_query": body.user_query, "mode": body.mode},
                config=config, # type: ignore[call-arg]
                version="v2",  
            ):
                kind: str = event["event"]
                metadata: dict = event.get("metadata", {})
                node_name: str = metadata.get("langgraph_node", "")
                checkpoint_id: Optional[str] = metadata.get("checkpoint_id")

                logger.debug("Event: %s | Node: %s | Metadata: %s | Checkpoint ID: %s", kind, node_name, metadata, checkpoint_id)

                # ── on_chain_start — verbose thinking steps ───────────────────

                if kind == "on_chain_start":

                    # Worker: fanout fires it without checkpoint_ns — counter it regardless
                    if node_name == "worker":
                        worker_count += 1
                        yield sse({"type": "verbose", "content": f"Writing section {worker_count}..."})
                        continue

                    # Skip the pregel outer duplicate for all other nodes
                    if not is_inner_node_event(event): #type: ignore
                        continue

                    # Skip nodes we have no label for (subgraph wrappers, output parsers, etc.)
                    if node_name in seen_verbose or node_name not in VERBOSE_NODE_LABELS:
                        continue

                    seen_verbose.add(node_name)
                    yield sse({"type": "verbose", "content": VERBOSE_NODE_LABELS[node_name]})

                # ── on_chat_model_start — "Thinking..." only for answer nodes ──
                elif kind == "on_chat_model_start":
                    # Guard: internal pipeline LLMs (orchestrator, worker, etc.)
                    # also fire this event — only surface it for user-facing nodes.
                    if node_name in ANSWER_NODES:
                        # print({"type": "verbose", "content": "Thinking..."})
                        yield sse({"type": "verbose", "content": "Understanding request..."})

                # ── on_tool_start — tool call initiated ───────────────────────
                elif kind == "on_tool_start":
                    tool_name: str = event.get("name", "tool")
                    tool_input: dict = event["data"].get("input", {})
                    summary = json.dumps(tool_input)[:120]
                    # print({"type": "verbose", "content": f"Calling tool: {tool_name}...", "detail": summary})
                    yield sse({"type": "verbose", "content": f"Calling tool: {tool_name}...", "detail": summary})

                # ── on_tool_end — tool call finished ──────────────────────────
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    # print({"type": "verbose", "content": f"Tool done: {tool_name}, processing result..."})
                    yield sse({"type": "verbose", "content": f"Tool done: {tool_name}, processing result..."})

                # ── on_chat_model_stream — token-level streaming ───────────────
                elif kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    content = chunk.content if chunk else None

                    if content and node_name:
                        if node_name in ANSWER_NODES:
                            # print({"type": "token", "content": content})
                            yield sse({"type": "token", "content": content})
                        elif node_name in REASONING_NODES:
                            # print({"type": "reasoning", "content": content})
                            yield sse({"type": "reasoning", "content": content})
                        # else: tool call internals / subgraph routing — skip

                # ── on_chain_end (root graph) — capture final state ────────────
                # Fix: this MUST be a top-level elif, not nested inside
                # on_chat_model_stream. kind cannot be two values simultaneously.
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output", {})

            # ── Post-stream: build and emit result event ───────────────────────
            assistant_response: str | None = (
                final_state["messages"][-1].content
                if final_state and final_state.get("messages")
                else None
            )

            assistant_response_blog: str | None = (
                final_state.get("final_blog")
                if final_state and body.mode == "generate"
                else None
            )

            latest_state = await agent.blog_agentic_ai.aget_state(config) #type: ignore[call-arg]
            captured_checkpoint_id: Optional[str] = (
                latest_state.config.get("configurable", {}).get("checkpoint_id")
                if latest_state else None
            )
            logger.info("checkpoint_id from aget_state: %s", captured_checkpoint_id)

            yield sse({
                "type":       "result",
                "response":   assistant_response,
                "final_blog": assistant_response_blog,
                "checkpoint_id": captured_checkpoint_id,
            })

            # ── Persist conversation to DB ─────────────────────────────────────
            try:
                db_response = await save_conversation_func(
                    user_id=user_id,
                    thread_id=thread_id,
                    user_query=body.user_query,  # type: ignore[arg-type]
                    assistant_response=assistant_response,
                    assistant_response_blog=assistant_response_blog,
                    checkpoint_id=captured_checkpoint_id
                )
                if not db_response.get("success"):
                    logger.error("DB save failed for thread %s: %s", thread_id, db_response)
                    yield sse({"type": "error", "detail": "Failed to save conversation"})
            except Exception as db_exc:
                # DB failure must not kill an otherwise successful stream
                logger.exception("DB save raised for thread %s", thread_id)
                yield sse({"type": "error", "detail": f"DB error: {db_exc}"})

        except asyncio.CancelledError:
            # Client disconnected mid-stream — not an application error
            logger.info("Client disconnected mid-stream (thread=%s)", thread_id)
            return

        except Exception as exc:
            logger.exception("Stream crashed!")
            
            # Even though it crashed, let's try to grab the last state 
            # and save it so the user doesn't lose everything.
            error_state = await agent.blog_agentic_ai.aget_state(config) #type: ignore[call-arg]

            error_captured_checkpoint_id: Optional[str] = (
                error_state.config.get("configurable", {}).get("checkpoint_id")
                if error_state else None
            )
            
            await save_conversation_func(
                user_id=user_id,
                thread_id=thread_id,
                user_query=body.user_query, # type: ignore[arg-type]
                assistant_response="[System Error: Interrupted]", 
                assistant_response_blog=None,
                checkpoint_id=error_captured_checkpoint_id
            )
            
            # CORRECT — frontend gets the checkpoint to retry from
            yield sse({
                "type":        "error",
                "detail":      "I ran into a problem, but I've saved our progress.",
                "retry_checkpoint_id": error_captured_checkpoint_id,  # ← add this
                "thread_id":   thread_id,
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/retry/{thread_id}/{checkpoint_id}")
async def retry_from_checkpoint(request: Request):

    checkpoint_id: str = "1f1213ce-913c-6086-802e-39747f82708d"
    # checkpoint_id: str = request.path_params.get("checkpoint_id") # type: ignore[call-arg]
    thread_id: str = request.path_params.get("thread_id") # type: ignore[call-arg]
    user_id: str = request.state.user.get("sub")

    return StreamingResponse(agent.blog_agentic_ai.astream_events(
        {"mode": "chat"},
        config={"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}, # type: ignore[call-arg]
        version="v2",
    ), media_type="text/event-stream")


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
            "user":      response,
            "jwt_token": response["jwt_token"],
            "success":   True,
            "message":   "User registered successfully",
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
            "user":      response,
            "jwt_token": response["jwt_token"],
            "success":   True,
            "message":   "User verified successfully",
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
# Text to Speech endpoint (optional, can be implemented later)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/tts/{thread_id}")
async def text_to_speech(request: Request):
    body_json = await request.json()
    raw_text: str = body_json.get("text", "").strip()
    voice: str    = body_json.get("voice", "troy")

    if not raw_text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No text provided")

    # ── Strip markdown before sending to Groq ──
    clean_text = strip_markdown(raw_text)

    if not clean_text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No speakable text after stripping markdown")

    clean_text = truncate_to_limit(clean_text)

    logger.info("tts: original_len=%d clean_len=%d", len(raw_text), len(clean_text))
    
    def audio_stream():
        with text_to_speech_model.audio.speech.with_streaming_response.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice,
            input=clean_text,   # ← clean text, not raw markdown
            response_format="wav",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache"},
    )