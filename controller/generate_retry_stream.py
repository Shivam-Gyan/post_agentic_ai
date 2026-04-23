from typing import Optional
import agent
import json
from controller.conversation_controller import save_conversation_func, save_retry_version, save_edit_turn
import logging
import asyncio
from langchain_core.messages import AIMessage
from uuid import uuid4

from states import BlogState

logger = logging.getLogger(__name__)

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



async def retry_func(thread_id: str):
    """Endpoint to retry a generation from a specific checkpoint."""
    # This function would extract the checkpoint_id and other necessary info from the request,
    # then call _run_agent_stream with that checkpoint_id to resume the generation.
    final_state  = list( agent.blog_agentic_ai.get_state_history({"configurable": {'thread_id': thread_id}})) # type: ignore
    # logger.info("Retrying from checkpoint_id %s for thread_id %s ", checkpoint_id, thread_id)

    # user_query:str | None = "arigato sensei"
    # # user_query:str | None = final_state.values.get('user_query') if final_state else None
    # mode = final_state.values.get('mode') if final_state else "chat"
    
    # inputs: BlogState | None = BlogState(user_query=user_query, mode=mode, messages=[]) if user_query and mode else None
    # response = await agent.blog_agentic_ai.ainvoke(
    #     inputs, # Often no new inputs are needed for a retry, but this can be
    #     config = {"configurable": {'thread_id': thread_id, "checkpoint_id": checkpoint_id}},

    # )
    # logger.info("Raw response from retry_func for user_query %s:", user_query)

    # logger.info("Retry response for checkpoint_id %s: %s", checkpoint_id, response['messages'][-1].content if response and response.get('messages') else "No response")

    # return response
    return final_state


#  run_stream_turn cycle of reqiest and response

async def _run_agent_stream(
    user_id: str,
    thread_id: str,
    mode: str,
    user_query: str,
    previous_final_checkpoint_id: str | None = None,
):
    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            **({"checkpoint_id": previous_final_checkpoint_id} if previous_final_checkpoint_id else {})
        }
    }
    # ✅ Separate clean config for history queries — never includes checkpoint_id
    base_config: dict = {"configurable": {"thread_id": thread_id}}

    inputs = {"user_query": user_query, "mode": mode}

    final_state: dict | None = None
    seen_verbose: set[str] = set()
    worker_count: int = 0

    # ✅ Defined OUTSIDE try so except block can always access it safely
    MODE_NODES = {
        "chat": ["conversation_subgraph", "chat_node"],
        "generate": ["orchestrator", "router_node", "research_node"],
        "refine": ["refine_subgraph", "refine_node"]
    }
    target_nodes = MODE_NODES.get(mode.lower(), ["conversation_subgraph", "chat_node"])

    edit_checkpoint_id = None
    retry_checkpoint_id = None
    final_checkpoint_id = None

    try:
        async for event in agent.blog_agentic_ai.astream_events(
            inputs,
            config=config, #type: ignore
            version="v2",
        ):
            kind = event["event"]
            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node", "")

            logger.info("Event kind=%s || node=%s", kind, node_name)

            if kind == "on_chain_start":
                if node_name == "worker":
                    worker_count += 1
                    yield sse({"type": "verbose", "content": f"Writing section {worker_count}..."})
                    continue
                if not is_inner_node_event(event) or node_name in seen_verbose or node_name not in VERBOSE_NODE_LABELS: #type: ignore
                    continue
                seen_verbose.add(node_name)
                yield sse({"type": "verbose", "content": VERBOSE_NODE_LABELS[node_name]})

            elif kind == "on_chat_model_start" and node_name in ANSWER_NODES:
                yield sse({"type": "verbose", "content": "Architecting response..."})

            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                tool_input = event["data"].get("input", {})
                yield sse({"type": "verbose", "content": f"Calling tool: {tool_name}...", "detail": json.dumps(tool_input)[:120]})

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                content = chunk.content if chunk else None
                if content and node_name:
                    if node_name in ANSWER_NODES:
                        yield sse({"type": "token", "content": content})
                    elif node_name in REASONING_NODES:
                        yield sse({"type": "reasoning", "content": content})

            elif kind == "on_chain_end" and event.get("name") == "LangGraph" and node_name == "":
                final_state = event["data"].get("output", {})

        # --- POST-STREAM PERSISTENCE ---
        # ✅ Always use base_config here, never config (which may have checkpoint_id)
        history = [s async for s in agent.blog_agentic_ai.aget_state_history(base_config, limit=20)] #type: ignore

        if not history:
            return

        final_checkpoint_id = history[0].config.get("configurable", {}).get("checkpoint_id")

        for snapshot in history:
            if not snapshot.config:
                continue
            if any(node in snapshot.next for node in target_nodes) and not retry_checkpoint_id:
                retry_checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
            if snapshot.metadata and snapshot.metadata.get("source") == "input":
                edit_checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
                break

        edit_checkpoint_id = edit_checkpoint_id or retry_checkpoint_id

        # ✅ Fixed
        assistant_response: str | None = next(
            (m.content for m in reversed((final_state or {}).get("messages", [])) if isinstance(m, AIMessage)), #type: ignore
            None
        )

        assistant_response_blog: str | None = (
            final_state.get("final_blog") if final_state and mode == "generate" else None
        )

        yield sse({
            "type": "result",
            "response": assistant_response,
            "final_blog": assistant_response_blog,
            "edit_id": edit_checkpoint_id,
            "retry_id": retry_checkpoint_id,
            "checkpoint_id": final_checkpoint_id,
        })

        try:
            db_res = await save_conversation_func(
                user_id=user_id,
                thread_id=thread_id,
                user_query=str(user_query),
                assistant_response=assistant_response,
                assistant_response_blog=assistant_response_blog,
                edit_checkpoint_id=edit_checkpoint_id,
                retry_checkpoint_id=retry_checkpoint_id,
                final_checkpoint_id=final_checkpoint_id,
            )
            if not db_res.get("success"):
                logger.error("DB Save failed: %s", db_res.get("message"))
        except Exception:
            logger.exception("Critical DB failure on success path")

    except (asyncio.CancelledError, Exception) as exc:
        is_cancel = isinstance(exc, asyncio.CancelledError)
        logger.warning("Stream %s for thread=%s", "cancelled" if is_cancel else "failed", thread_id)

        try:
            history = [s async for s in agent.blog_agentic_ai.aget_state_history(base_config, limit=10)] #type: ignore
            if history:
                final_checkpoint_id = history[0].config.get("configurable", {}).get("checkpoint_id")
                for snapshot in history:
                    if any(node in snapshot.next for node in target_nodes) and not retry_checkpoint_id:
                        retry_checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
                    if snapshot.metadata and snapshot.metadata.get("source") == "input":
                        edit_checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
                        break
        except Exception:
            logger.error("Failed to fetch history during error handling")

        if is_cancel:
            # User stopped — partial state is valid, retry/edit makes sense
            yield sse({
                "type": "error",
                "error": "Stopped by user",
                "edit_id": edit_checkpoint_id,
                "retry_id": retry_checkpoint_id,
            })

            db_res = await save_conversation_func(
                user_id=user_id,
                thread_id=thread_id,
                user_query=str(user_query),
                assistant_response="[Stopped by user]",
                assistant_response_blog=None,
                edit_checkpoint_id=edit_checkpoint_id,
                retry_checkpoint_id=retry_checkpoint_id,
                final_checkpoint_id=final_checkpoint_id,  # valid — graph was mid-run, not broken
            )
            if not db_res.get("success"):
                logger.error("DB Save failed on cancel: %s", db_res.get("message"))

            raise  # re-raise CancelledError so server closes the stream

        else:
            # Internal error — checkpoint state is unreliable
            # Send edit_id only — user can re-ask, but NOT resume from broken state
            yield sse({
                "type": "error",
                "error": str(exc),
                "edit_id": edit_checkpoint_id,   # ✅ re-ask from scratch
                "retry_id": None,                # ❌ do NOT offer retry — state is unreliable
            })

            db_res = await save_conversation_func(
                user_id=user_id,
                thread_id=thread_id,
                user_query=str(user_query),
                assistant_response="[Error: Something went wrong]",
                assistant_response_blog=None,
                edit_checkpoint_id=edit_checkpoint_id,
                retry_checkpoint_id=None,   # ← None means "retry not available for this message"
                final_checkpoint_id=None,        # ❌ do NOT save broken checkpoint as resumable
            )
            if not db_res.get("success"):
                logger.error("DB Save failed on error: %s", db_res.get("message"))


async def _run_agent_stream_retry(
    user_id: str,
    thread_id: str,
    mode: str,
    retry_checkpoint_id: str,
    user_query: str,
):
    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": retry_checkpoint_id,
        }
    }
    base_config: dict = {"configurable": {"thread_id": thread_id}}

    # Bust checkpoint cache so LLM actually re-executes
    await agent.blog_agentic_ai.aupdate_state(
        config, #type: ignore
        {"retry_seed": str(uuid4())},
    )
    await asyncio.sleep(0)

    state_after = await agent.blog_agentic_ai.aget_state(base_config) #type: ignore
    new_checkpoint_id = state_after.config.get("configurable", {}).get("checkpoint_id")

    # ✅ Keep only this one — useful for production debugging
    logger.info("Retry cache-busted for thread=%s new_checkpoint=%s", thread_id, new_checkpoint_id)

    replay_config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": new_checkpoint_id,
        }
    }

    inputs = None
    final_state: dict | None = None
    seen_verbose: set[str] = set()
    worker_count: int = 0

    MODE_NODES = {
        "chat": ["conversation_subgraph", "chat_node"],
        "generate": ["orchestrator", "router_node", "research_node"],
        "refine": ["refine_subgraph", "refine_node"]
    }
    target_nodes = MODE_NODES.get(mode.lower(), ["conversation_subgraph", "chat_node"])
    new_final_checkpoint_id = None

    try:
        async for event in agent.blog_agentic_ai.astream_events(
            inputs,
            config=replay_config,  # type: ignore
            version="v2",
        ):
            kind = event["event"]
            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node", "")

            if kind == "on_chain_start":
                if node_name == "worker":
                    worker_count += 1
                    yield sse({"type": "verbose", "content": f"Writing section {worker_count}..."})
                    continue
                if not is_inner_node_event(event) or node_name in seen_verbose or node_name not in VERBOSE_NODE_LABELS: #type: ignore
                    continue
                seen_verbose.add(node_name)
                yield sse({"type": "verbose", "content": VERBOSE_NODE_LABELS[node_name]})

            elif kind == "on_chat_model_start" and node_name in ANSWER_NODES:
                logger.info("🔥 LLM actually called during retry for node=%s", node_name)
                yield sse({"type": "verbose", "content": "Architecting response..."})

            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                tool_input = event["data"].get("input", {})
                yield sse({"type": "verbose", "content": f"Calling tool: {tool_name}...", "detail": json.dumps(tool_input)[:120]})

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                content = chunk.content if chunk else None
                if content and node_name:
                    if node_name in ANSWER_NODES:
                        yield sse({"type": "token", "content": content})
                    elif node_name in REASONING_NODES:
                        yield sse({"type": "reasoning", "content": content})

            elif kind == "on_chain_end" and event.get("name") == "LangGraph" and node_name == "":
                final_state = event["data"].get("output", {})

        # --- POST-STREAM ---
        history = [s async for s in agent.blog_agentic_ai.aget_state_history(base_config, limit=5)] #type: ignore

        if not history:
            return

        new_final_checkpoint_id = history[0].config.get("configurable", {}).get("checkpoint_id")

        assistant_response: str | None = next(
            (m.content for m in reversed((final_state or {}).get("messages", [])) if isinstance(m, AIMessage)), #type: ignore
            None
        )
        assistant_response_blog: str | None = (
            final_state.get("final_blog") if final_state and mode == "generate" else None
        )

        yield sse({
            "type": "result",
            "response": assistant_response,
            "final_blog": assistant_response_blog,
            "checkpoint_id": new_final_checkpoint_id,
        })

        try:
            db_res = await save_retry_version(
                user_id=user_id,
                thread_id=thread_id,
                assistant_response=assistant_response,
                assistant_response_blog=assistant_response_blog,
                final_checkpoint_id=new_final_checkpoint_id,
                retry_checkpoint_id=retry_checkpoint_id,
            )
            if not db_res.get("success"):
                logger.error("DB Save (retry) failed: %s", db_res.get("message"))
        except Exception:
            logger.exception("Critical DB failure on retry success path")

    except (asyncio.CancelledError, Exception) as exc:
        is_cancel = isinstance(exc, asyncio.CancelledError)
        logger.warning("Retry stream %s for thread=%s", "cancelled" if is_cancel else "failed", thread_id)

        try:
            history = [s async for s in agent.blog_agentic_ai.aget_state_history(base_config, limit=5)] #type: ignore
            if history:
                new_final_checkpoint_id = history[0].config.get("configurable", {}).get("checkpoint_id")
        except Exception:
            logger.error("Failed to fetch history during retry error handling")

        if is_cancel:
            yield sse({"type": "error", "error": "Stopped by user"})
            await save_retry_version(
                user_id=user_id,
                thread_id=thread_id,
                assistant_response="[Stopped by user]",
                assistant_response_blog=None,
                final_checkpoint_id=new_final_checkpoint_id,
                retry_checkpoint_id=retry_checkpoint_id,
            )
            raise
        else:
            yield sse({"type": "error", "error": str(exc)})
            logger.error("Retry failed with internal error for thread=%s, no new version saved", thread_id)



# edit and retry 
async def _run_agent_stream_edit(
    user_id: str,
    thread_id: str,
    mode: str,
    edit_checkpoint_id: str,  # points to source="input" — before user message
    new_user_query: str,       # the edited/new query
):
    # ✅ Rewind to edit point — this is BEFORE the original user message
    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": edit_checkpoint_id,
        }
    }
    base_config: dict = {"configurable": {"thread_id": thread_id}}

    # ✅ Pass new inputs — this replaces the original user message entirely
    # Unlike retry (inputs=None), edit MUST send new inputs
    # because we're starting fresh from before the user message
    inputs = {"user_query": new_user_query, "mode": mode}

    final_state: dict | None = None
    seen_verbose: set[str] = set()
    worker_count: int = 0

    MODE_NODES = {
        "chat": ["conversation_subgraph", "chat_node"],
        "generate": ["orchestrator", "router_node", "research_node"],
        "refine": ["refine_subgraph", "refine_node"]
    }
    target_nodes = MODE_NODES.get(mode.lower(), ["conversation_subgraph", "chat_node"])

    edit_checkpoint_id_new = None
    retry_checkpoint_id_new = None
    final_checkpoint_id_new = None

    try:
        async for event in agent.blog_agentic_ai.astream_events(
            inputs,
            config=config,  #type: ignore
            version="v2",
        ):
            kind = event["event"]
            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node", "")

            if kind == "on_chain_start":
                if node_name == "worker":
                    worker_count += 1
                    yield sse({"type": "verbose", "content": f"Writing section {worker_count}..."})
                    continue
                if not is_inner_node_event(event) or node_name in seen_verbose or node_name not in VERBOSE_NODE_LABELS: #type: ignore
                    continue
                seen_verbose.add(node_name)
                yield sse({"type": "verbose", "content": VERBOSE_NODE_LABELS[node_name]})

            elif kind == "on_chat_model_start" and node_name in ANSWER_NODES:
                yield sse({"type": "verbose", "content": "Architecting response..."})

            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                tool_input = event["data"].get("input", {})
                yield sse({"type": "verbose", "content": f"Calling tool: {tool_name}...", "detail": json.dumps(tool_input)[:120]})

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                content = chunk.content if chunk else None
                if content and node_name:
                    if node_name in ANSWER_NODES:
                        yield sse({"type": "token", "content": content})
                    elif node_name in REASONING_NODES:
                        yield sse({"type": "reasoning", "content": content})

            elif kind == "on_chain_end" and event.get("name") == "LangGraph" and node_name == "":
                final_state = event["data"].get("output", {})

        # --- POST-STREAM PERSISTENCE ---
        # ✅ Same checkpoint extraction as normal generation turn
        history = [s async for s in agent.blog_agentic_ai.aget_state_history(base_config, limit=20)] #type: ignore

        if not history:
            return

        final_checkpoint_id_new = history[0].config.get("configurable", {}).get("checkpoint_id")

        for snapshot in history:
            if not snapshot.config:
                continue
            if any(node in snapshot.next for node in target_nodes) and not retry_checkpoint_id_new:
                retry_checkpoint_id_new = snapshot.config.get("configurable", {}).get("checkpoint_id")
            if snapshot.metadata and snapshot.metadata.get("source") == "input":
                edit_checkpoint_id_new = snapshot.config.get("configurable", {}).get("checkpoint_id")
                break

        edit_checkpoint_id_new = edit_checkpoint_id_new or retry_checkpoint_id_new

        assistant_response: str | None = next(
            (m.content for m in reversed((final_state or {}).get("messages", [])) if isinstance(m, AIMessage)), #type: ignore
            None
        )
        assistant_response_blog: str | None = (
            final_state.get("final_blog") if final_state and mode == "generate" else None
        )

        # ✅ Same SSE shape as normal generation — frontend treats this like a fresh turn
        yield sse({
            "type": "result",
            "response": assistant_response,
            "final_blog": assistant_response_blog,
            "edit_id": edit_checkpoint_id_new,      # new edit_id for the edited user message
            "retry_id": retry_checkpoint_id_new,    # new retry_id for the new assistant message
            "checkpoint_id": final_checkpoint_id_new,
        })

        try:
            db_res = await save_edit_turn(
                user_id=user_id,
                thread_id=thread_id,
                new_user_query=new_user_query,
                assistant_response=assistant_response,
                assistant_response_blog=assistant_response_blog,
                edit_checkpoint_id_new=edit_checkpoint_id_new,
                edit_checkpoint_id=edit_checkpoint_id,
                retry_checkpoint_id=retry_checkpoint_id_new,
                final_checkpoint_id=final_checkpoint_id_new,
            )
            if not db_res.get("success"):
                logger.error("DB Save (edit) failed: %s", db_res.get("message"))
        except Exception:
            logger.exception("Critical DB failure on edit success path")

    except (asyncio.CancelledError, Exception) as exc:
        is_cancel = isinstance(exc, asyncio.CancelledError)
        logger.warning("Edit stream %s for thread=%s", "cancelled" if is_cancel else "failed", thread_id)

        try:
            history = [s async for s in agent.blog_agentic_ai.aget_state_history(base_config, limit=10)] #type: ignore
            if history:
                final_checkpoint_id_new = history[0].config.get("configurable", {}).get("checkpoint_id")
                for snapshot in history:
                    if any(node in snapshot.next for node in target_nodes) and not retry_checkpoint_id_new:
                        retry_checkpoint_id_new = snapshot.config.get("configurable", {}).get("checkpoint_id")
                    if snapshot.metadata and snapshot.metadata.get("source") == "input":
                        edit_checkpoint_id_new = snapshot.config.get("configurable", {}).get("checkpoint_id")
                        break
        except Exception:
            logger.error("Failed to fetch history during edit error handling")

        if is_cancel:
            yield sse({
                "type": "error",
                "error": "Stopped by user",
                "edit_id": edit_checkpoint_id_new,
                "retry_id": retry_checkpoint_id_new,
            })
            await save_edit_turn(
                user_id=user_id,
                thread_id=thread_id,
                new_user_query=new_user_query,
                assistant_response="[Stopped by user]",
                assistant_response_blog=None,
                edit_checkpoint_id_new=edit_checkpoint_id_new,
                edit_checkpoint_id=edit_checkpoint_id,
                retry_checkpoint_id=retry_checkpoint_id_new,
                final_checkpoint_id=final_checkpoint_id_new,
            )
            raise
        else:
            yield sse({
                "type": "error",
                "error": str(exc),
                "edit_id": edit_checkpoint_id_new,
                "retry_id": None,
            })
            await save_edit_turn(
                user_id=user_id,
                thread_id=thread_id,
                new_user_query=new_user_query,
                assistant_response="[Error: Something went wrong]",
                assistant_response_blog=None,
                edit_checkpoint_id_new=edit_checkpoint_id_new,
                edit_checkpoint_id=edit_checkpoint_id,
                retry_checkpoint_id=None,
                final_checkpoint_id=None,
            )
