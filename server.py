
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from nodes import detail_node, fanout, generate_blog_plan, reducer, worker
from states import BlogState

# 1. initialize the state graph
graph = StateGraph(BlogState)

# 2. define the nodes
graph.add_node('generate_blog_plan', generate_blog_plan)
graph.add_node('worker', worker) #type: ignore
graph.add_node('reducer',reducer)
graph.add_node('detail_node', detail_node)

# 3. define the edges
graph.add_edge(START,'detail_node')
graph.add_edge('detail_node', 'generate_blog_plan')
graph.add_conditional_edges('generate_blog_plan',fanout,['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)
checkpointer = MemorySaver()


# 5. compile the graph
blog_agentic_ai = graph.compile(checkpointer=checkpointer) 



async def main():
    print("\n----------- Agentic AI Blog Generator ----------\n")

    user_input= input("Please enter the blog description: ")
    print(f"\nUser : {user_input}\n")

    initial_state = BlogState(
        blog_description=user_input,
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
        print(final_state["final_blog"])

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")