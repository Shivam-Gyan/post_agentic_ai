from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent import blog_agentic_ai
from typing import Optional
from fastapi import HTTPException, status
from pydantic import BaseModel

app = FastAPI(title="Blog Agentic AI")

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

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Start a blog generation/refinement/chat turn and return the final state.

    Behavior:
    - If `mode` is provided, prefix `user_query` with `<mode>:` to route the graph.
    - Otherwise use `user_query` as-is.
    """
    # Build user_query for the graph's intent detection node
    if req.mode and req.user_query:
        user_query = f"{req.mode}: {req.user_query}"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide `blog_description` or `user_query`")

    config = {
        "configurable": {
            "thread_id": req.thread_id or "blog_generation_thread",
        }
    }

    try:
        final_state = await blog_agentic_ai.ainvoke({"user_query": user_query}, config=config)  # type: ignore

        # Convert pydantic models / objects to plain dict where needed
        result = {
            "mode": final_state.get("mode"),
            "response": final_state['messages'][-1].content if final_state.get("messages") else None,
            "final_blog": final_state.get("final_blog"),
            # "messages": [m.content for m in final_state.get("messages", [])] if final_state.get("messages") else [],
            # "summary": final_state.get("summary"),
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")




