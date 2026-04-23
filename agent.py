
import asyncio
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from database.mongodb_checkpointer import get_checkpointer
from node.conversation_nodes import chat_node_func
from node.generate_nodes import router_node,research_node, router_condition_func, fanout, orchestrator, reducer, worker, publish_node
from node.refinement_nodes import refine_node_func, refine_structured_output_model
from node.intent_detection_nodes import intent_detection_node, intent_router_func
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import tools_condition, ToolNode
from config_mcp_server import SERVERS
from states import BlogState
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "blog_agentic_ai")

# refinement subgraph to handle the refinement 
async def build_refine_subgraph():
    g = StateGraph(BlogState)

    # define the nodes for refinement subgraph
    g.add_node('refine_structured_output_node', refine_structured_output_model)
    g.add_node('refine_node', refine_node_func)

    # define the edges for refinement subgraph
    g.add_edge(START,'refine_structured_output_node')
    g.add_edge('refine_structured_output_node','refine_node')
    # g.set_entry_point('refine_structured_output_node')
    g.add_edge('refine_node',END)

    # No separate checkpointer — the parent graph's MemorySaver handles
    # state persistence so refinement history carries over between turns.
    return g.compile()# conversation subgraph to handle the conversation 



async def build_conversation_subgraph():

    mcp_client  = MultiServerMCPClient(SERVERS) #type: ignore

    global tools
    try:
        tools = await mcp_client.get_tools()
        # print(f"Loaded tools from MCP servers: {tools}")
    except Exception as e:
        # If MCP servers can't be contacted at startup (common in dev),
        # fall back to an empty tools mapping and continue — the graph
        # can still run in closed-book/chat modes.
        print(f"Warning: failed to load MCP tools: {e}")
        tools = {}

    # `tools` shape may not exactly match the static type expected by ToolNode;
    # silence precise arg-type checking here since runtime value is correct.
    tools_node = ToolNode(tools = tools)  # type: ignore[arg-type]


    g = StateGraph(BlogState)

    # define the nodes for conversation subgraph
    g.add_node('chat_node', chat_node_func)
    g.add_node('tools',tools_node) # neccessary to name as it 'tools'

    # define the edges for conversation subgraph
    g.add_edge(START,'chat_node')
    g.add_conditional_edges('chat_node', tools_condition)
    g.add_edge('tools','chat_node')
    # g.add_edge('chat_node',END)
    
    # No separate checkpointer — the parent graph's MemorySaver handles
    # state persistence so AIMessages carry over between turns.
    return g.compile()


# main graph of Agentic AI Blog Generator
async def build_blog_graph():
    """Build and compile the blog generation graph.
    Extracted into a function so Streamlit can cache it with @st.cache_resource
    and keep the MemorySaver alive across reruns / file-watcher reloads."""

    refine_subgraph = await build_refine_subgraph()
    conversation_subgraph = await build_conversation_subgraph()
    # 1. initialize the state graph
    g = StateGraph(BlogState)

    # 2. define the nodes
    g.add_node('intent_node', intent_detection_node)
    g.add_node('refine_subgraph', refine_subgraph)
    g.add_node('conversation_subgraph', conversation_subgraph)
    g.add_node('router_node', router_node)
    g.add_node('research_node', research_node)
    g.add_node('orchestrator', orchestrator)
    g.add_node('worker', worker) #type: ignore
    g.add_node('reducer',reducer)
    # g.add_node('publish_node', publish_node)

    # 3. define the edges
    g.add_edge(START,'intent_node')
    # g.add_edge('intent_node',END)
    g.add_conditional_edges('intent_node', intent_router_func) #type: ignore
    g.add_edge('refine_subgraph',END)
    g.add_edge('conversation_subgraph',END)
    g.add_conditional_edges('router_node',router_condition_func )
    g.add_edge('research_node', 'orchestrator')
    g.add_conditional_edges('orchestrator',fanout,['worker'])
    g.add_edge('worker', 'reducer')
    g.add_edge('reducer', END)
    # g.add_edge('reducer', 'publish_node')
    # g.add_edge('publish_node', END)

    checkpointer = get_checkpointer()
    return g.compile(checkpointer=checkpointer)


_init_lock = asyncio.Lock()
blog_agentic_ai = None

async def init_blog_graph():
    global blog_agentic_ai
    async with _init_lock:              # only one coroutine enters at a time
        if blog_agentic_ai is None:     # re-check inside the lock
            blog_agentic_ai = await build_blog_graph()
    return blog_agentic_ai



# # agent.py
# # 
# # KEY CHANGES vs your original:
# #
# # 1. _mcp_client and _mcp_tools are module-level singletons.
# #    MultiServerMCPClient.__aenter__() is called ONCE in init_mcp_tools(),
# #    which keeps the stdio subprocess alive for the entire app lifetime.
# #    Previously, langchain_mcp_adapters was spawning a NEW subprocess on
# #    every single tool call — that subprocess creation hits anyio's checkpoint
# #    (await sleep(0)) which gets cancelled inside astream_events' task group.
# #
# # 2. build_conversation_subgraph() no longer creates its own MCP client.
# #    It just calls init_mcp_tools() which returns the already-loaded list.
# #
# # 3. init_blog_graph() now calls init_mcp_tools() before building the graph
# #    so tools are guaranteed ready before any node tries to use them.
# #
# # 4. shutdown_mcp() is exposed so main.py lifespan can cleanly close the
# #    stdio subprocess on app shutdown.

# import asyncio
# import os
# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
# from database.mongodb_checkpointer import get_checkpointer
# from node.conversation_nodes import chat_node_func
# from node.generate_nodes import (
#     router_node, research_node, router_condition_func,
#     fanout, orchestrator, reducer, worker, publish_node
# )
# from node.refinement_nodes import refine_node_func, refine_structured_output_model
# from node.intent_detection_nodes import intent_detection_node, intent_router_func
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import tools_condition, ToolNode
# from config_mcp_server import SERVERS
# from states import BlogState
# from dotenv import load_dotenv

# load_dotenv()

# os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "blog_agentic_ai")

# import logging
# logger = logging.getLogger(__name__)

# # ──────────────────────────────────────────────────────────────────────────────
# # MCP singleton — one subprocess for the entire app lifetime
# # ──────────────────────────────────────────────────────────────────────────────

# _mcp_client: MultiServerMCPClient | None = None
# _mcp_tools: list | None = None
# _mcp_lock = asyncio.Lock()


# async def init_mcp_tools() -> list:
#     """
#     Initialize the MCP client exactly once and keep the stdio subprocess alive.
    
#     Why this matters:
#     - stdio transport spawns a subprocess via anyio.open_process()
#     - anyio always calls checkpoint() (await sleep(0)) during process creation
#     - Inside astream_events, LangGraph wraps nodes in asyncio.create_task()
#     - Any awaitable that hits a checkpoint inside a cancelled task raises CancelledError
#     - Solution: spawn the subprocess BEFORE any streaming context exists,
#       then reuse the already-open connection for every tool call
#     """
#     global _mcp_client, _mcp_tools

#     if _mcp_tools is not None:
#         return _mcp_tools

#     async with _mcp_lock:
#         # Double-check inside lock — another coroutine may have initialized
#         # between the check above and acquiring the lock
#         if _mcp_tools is not None:
#             return _mcp_tools

#         try:
#             _mcp_client = MultiServerMCPClient(SERVERS)  # type: ignore
#             # __aenter__ starts the subprocess and opens stdin/stdout pipes.
#             # From this point on, tool calls just send messages over the pipe —
#             # no new subprocess, no anyio checkpoint, no CancelledError.
#             await _mcp_client.__aenter__()
#             _mcp_tools = await _mcp_client.get_tools()
#             logger.info("MCP tools loaded: %s", [t.name for t in _mcp_tools])
#         except Exception as e:
#             logger.warning("Failed to load MCP tools: %s", e)
#             # Fall back to empty list so the graph can still run in
#             # closed-book / chat modes without tools
#             _mcp_tools = []

#     return _mcp_tools


# async def shutdown_mcp() -> None:
#     """
#     Call this from FastAPI lifespan on shutdown to cleanly terminate
#     the stdio subprocess. Prevents zombie processes on server restart.
#     """
#     global _mcp_client
#     if _mcp_client is not None:
#         try:
#             await _mcp_client.__aexit__(None, None, None)
#             logger.info("MCP client shut down cleanly")
#         except Exception as e:
#             logger.warning("MCP shutdown error (non-fatal): %s", e)
#         finally:
#             _mcp_client = None


# # ──────────────────────────────────────────────────────────────────────────────
# # Refinement subgraph
# # ──────────────────────────────────────────────────────────────────────────────

# async def build_refine_subgraph():
#     g = StateGraph(BlogState)

#     g.add_node('refine_structured_output_node', refine_structured_output_model)
#     g.add_node('refine_node', refine_node_func)

#     g.add_edge(START, 'refine_structured_output_node')
#     g.add_edge('refine_structured_output_node', 'refine_node')
#     g.add_edge('refine_node', END)

#     # No separate checkpointer — parent graph's checkpointer handles persistence
#     return g.compile()


# # ──────────────────────────────────────────────────────────────────────────────
# # Conversation subgraph
# # ──────────────────────────────────────────────────────────────────────────────

# async def build_conversation_subgraph():
#     # Use the already-initialized cached tools — no new MCP client created here.
#     # This is the critical change: previously this function created its own
#     # MultiServerMCPClient which spawned a subprocess on every tool call.
#     tools = await init_mcp_tools()

#     tools_node = ToolNode(tools=tools)  # type: ignore[arg-type]

#     g = StateGraph(BlogState)

#     g.add_node('chat_node', chat_node_func)
#     g.add_node('tools', tools_node)  # must be named 'tools' for tools_condition

#     g.add_edge(START, 'chat_node')
#     g.add_conditional_edges('chat_node', tools_condition)
#     g.add_edge('tools', 'chat_node')

#     # No separate checkpointer — parent graph's checkpointer handles persistence
#     return g.compile()


# # ──────────────────────────────────────────────────────────────────────────────
# # Main blog graph
# # ──────────────────────────────────────────────────────────────────────────────

# async def build_blog_graph():
#     """
#     Build and compile the main blog generation graph.
#     MCP tools must already be initialized before this is called
#     (guaranteed by init_blog_graph() calling init_mcp_tools() first).
#     """
#     refine_subgraph = await build_refine_subgraph()
#     conversation_subgraph = await build_conversation_subgraph()

#     g = StateGraph(BlogState)

#     g.add_node('intent_node', intent_detection_node)
#     g.add_node('refine_subgraph', refine_subgraph)
#     g.add_node('conversation_subgraph', conversation_subgraph)
#     g.add_node('router_node', router_node)
#     g.add_node('research_node', research_node)
#     g.add_node('orchestrator', orchestrator)
#     g.add_node('worker', worker)  # type: ignore
#     g.add_node('reducer', reducer)

#     g.add_edge(START, 'intent_node')
#     g.add_conditional_edges('intent_node', intent_router_func)  # type: ignore
#     g.add_edge('refine_subgraph', END)
#     g.add_edge('conversation_subgraph', END)
#     g.add_conditional_edges('router_node', router_condition_func)
#     g.add_edge('research_node', 'orchestrator')
#     g.add_conditional_edges('orchestrator', fanout, ['worker'])
#     g.add_edge('worker', 'reducer')
#     g.add_edge('reducer', END)

#     checkpointer = get_checkpointer()
#     return g.compile(checkpointer=checkpointer)


# # ──────────────────────────────────────────────────────────────────────────────
# # Public init — called once from FastAPI lifespan
# # ──────────────────────────────────────────────────────────────────────────────

# _init_lock = asyncio.Lock()
# blog_agentic_ai = None


# async def init_blog_graph():
#     global blog_agentic_ai
#     async with _init_lock:
#         if blog_agentic_ai is None:
#             # Initialize MCP first — subprocess must be alive before the graph
#             # is built so ToolNode receives the actual tool list, not an empty one
#             await init_mcp_tools()
#             blog_agentic_ai = await build_blog_graph()
#     return blog_agentic_ai