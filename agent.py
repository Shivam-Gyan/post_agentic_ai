
import asyncio
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from database.mongodb_checkpointer import get_checkpointer
from node.conversation_nodes import chat_node_func
from node.generate_nodes import router_node,research_node, router_condition_func, fanout, orchestrator, reducer, worker, publish_node
from node.refinement_nodes import refine_node_func, refine_structured_output_model
from node.intent_detection_nodes import intent_detection_node, intent_router_func
from states import BlogState
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "blog_agentic_ai")

# refinement subgraph to handle the refinement 
def build_refine_subgraph():
    g = StateGraph(BlogState)

    # define the nodes for refinement subgraph
    g.add_node('refine_structured_output_node', refine_structured_output_model)
    g.add_node('refine_node', refine_node_func)

    # define the edges for refinement subgraph
    g.add_edge(START,'refine_structured_output_node')
    g.add_edge('refine_structured_output_node','refine_node')
    g.set_entry_point('refine_structured_output_node')
    g.add_edge('refine_node',END)

    # No separate checkpointer — the parent graph's MemorySaver handles
    # state persistence so refinement history carries over between turns.
    return g.compile()# conversation subgraph to handle the conversation 



def build_conversation_subgraph():
    g = StateGraph(BlogState)

    # define the nodes for conversation subgraph
    g.add_node('chat_node', chat_node_func)

    # define the edges for conversation subgraph
    g.add_edge(START,'chat_node')
    g.add_edge('chat_node',END)
    
    # No separate checkpointer — the parent graph's MemorySaver handles
    # state persistence so AIMessages carry over between turns.
    return g.compile()


# main graph of Agentic AI Blog Generator
def build_blog_graph():
    """Build and compile the blog generation graph.
    Extracted into a function so Streamlit can cache it with @st.cache_resource
    and keep the MemorySaver alive across reruns / file-watcher reloads."""

    refine_subgraph = build_refine_subgraph()
    conversation_subgraph = build_conversation_subgraph()
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


# Default module-level instance (used when running without Streamlit)
blog_agentic_ai = build_blog_graph()

# async def main():
#     print("\n----------- Agentic AI Blog Generator ----------\n")
#     print("Type 'exit' or 'quit' to stop.\n")

#     config = {
#         "configurable": {
#             "thread_id": "blog_generation_thread",
#         }
#     }

#     while True:
#         try:
#             user_input = input("You: ").strip()

#             if not user_input:
#                 continue

#             if user_input.lower() in ("exit", "quit"):
#                 print("\n👋 Goodbye!\n")
#                 break

#             print()

#             # Pass ONLY the fields that change per turn.
#             # Passing the full initial_state would overwrite plain (non-reducer)
#             # fields like `summary` with their blank defaults on every turn,
#             # wiping the checkpointed summary. Reducer fields like `messages`
#             # (add_messages) are safe either way, but plain fields are not.
#             final_state = await blog_agentic_ai.ainvoke(
#                 {"user_query": user_input}, # type: ignore
#                 config=config  # type: ignore
#             )

#             # print(blog_agentic_ai.get_state(config=config).values['messages']) #type: ignore

#             mode = final_state["mode"]
#             print(f"[Mode: {mode}]\n")

#             if mode == "guard":
#                 messages = final_state["messages"]
#                 if messages:
#                     print(f"Assistant: {messages[-1].content}\n")

#             if mode == "chat":
#                 messages = final_state["messages"]
#                 # print(f"\nConversation AI : ({messages[-1].content}):")
#                 if messages:
#                     print(f"Assistant: {messages[-1].content}\n")

#             elif mode in ("generate", "refine"):
#                 blog = final_state.get("final_blog", "")
#                 if blog:
#                     print(f"📝 Blog:\n\n{blog}\n")
#                 else:
#                     print("⚠️  No blog generated yet.\n")

#             elif mode == "publish":
#                 result = final_state.get("publish_result", "")
#                 print(f"🚀 Publish result: {result}\n")

#         except KeyboardInterrupt:
#             print("\n\n⚠️ Interrupted.")
#             break

#         except Exception as e:
#             print(f"\n❌ An error occurred: {e}\n")





# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n\n⚠️ Process interrupted by user.")
#     except Exception as e:
#         print(f"\n\n❌ Unexpected error: {e}")