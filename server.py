
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from node.generate_nodes import router_node,research_node, router_condition_func, fanout, orchestrator, reducer, worker, publish_node
from node.refinement_nodes import refine_node_func, refine_structured_output_model
from node.intent_detection_nodes import intent_detection_node, intent_router_func
from states import BlogState

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

    # compiling the subgraph
    cp = MemorySaver()
    return g.compile(checkpointer=cp)


# conversation subgraph to handle the conversation 
def build_conversation_subgraph():
    #logic here 
    pass


# main graph of Agentic AI Blog Generator
def build_blog_graph():
    """Build and compile the blog generation graph.
    Extracted into a function so Streamlit can cache it with @st.cache_resource
    and keep the MemorySaver alive across reruns / file-watcher reloads."""

    refine_subgraph = build_refine_subgraph()

    # 1. initialize the state graph
    g = StateGraph(BlogState)

    # 2. define the nodes
    g.add_node('intent_node', intent_detection_node)
    g.add_node('refine_subgraph', refine_subgraph)
    # g.add_node('orchestrator', orchestrator)
    # g.add_node('worker', worker) #type: ignore
    # g.add_node('reducer',reducer)
    # g.add_node('research_node', research_node)
    # g.add_node('router_node', router_node)
    # g.add_node('publish_node', publish_node)

    # 3. define the edges
    g.add_edge(START,'intent_node')
    g.add_conditional_edges('intent_node', intent_router_func) #type: ignore
    g.add_edge('refine_subgraph',END)
    # g.add_edge('intent_node',END)
    # g.add_edge('refine_structured_output_node',END)
    # g.add_edge(START,'router_node')
    # g.add_conditional_edges('router_node',router_condition_func )
    # g.add_edge('research_node', 'orchestrator')
    # g.add_conditional_edges('orchestrator',fanout,['worker'])
    # g.add_edge('worker', 'reducer')
    # g.add_edge('reducer', 'publish_node')
    # g.add_edge('publish_node', END)

    cp = MemorySaver()
    return g.compile(checkpointer=cp)


# Default module-level instance (used when running without Streamlit)
blog_agentic_ai = build_blog_graph()



async def main():
    print("\n----------- Agentic AI Blog Generator ----------\n")

    user_input= input("Please enter the blog description: ")
    print(f"\nUser : {user_input}\n")

    initial_state = BlogState(
        user_query=user_input,
        # blog_topic="Discovery of rocket science and its impact on modern space exploration"
    )

    config = {
        "configurable": {
            "thread_id": "blog_generation_thread",
        }
    }



    try:
        # 🔥 Async call (graph runs here)
        final_state = await blog_agentic_ai.ainvoke(initial_state, config=config)  # type: ignore

        print("\n\nFinal Blog Output:\n\n")
        # print(final_state["refinement"])
        print(final_state["mode"])
        print(final_state["user_query"])
        print(final_state["final_blog"])
        # print(final_state)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")