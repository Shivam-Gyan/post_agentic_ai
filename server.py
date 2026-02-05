
import threading
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from nodes import fanout, generate_blog_plan, reducer, worker
from states import BlogState
from utils import loading_animation

# 1. initialize the state graph
graph = StateGraph(BlogState)

# 2. define the nodes
graph.add_node('generate_blog_plan', generate_blog_plan)
graph.add_node('worker', worker) #type: ignore
graph.add_node('reducer',reducer)

# 3. define the edges
graph.add_edge(START,'generate_blog_plan')
graph.add_conditional_edges('generate_blog_plan',fanout,['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)
# 4. initialize checkpointer memory saver
checkpointer = MemorySaver()


# 5. compile the graph
blog_agentic_ai = graph.compile(checkpointer=checkpointer) 


if __name__ == "__main__":
    print("\n----------- Agentic AI Blog Generator ----------\n")

    initial_state = BlogState(
        blog_topic="The Future of Artificial Intelligence in Everyday Life"
    )

    config = {
        "configurable": {
            "thread_id": "blog_generation_thread",
        }
    }

    stop_event = threading.Event()

    loader_thread = threading.Thread(
        target=loading_animation,
        args=(stop_event,),
        daemon=True
    )

    loader_thread.start()

    # 🔥 Blocking call (graph runs here)
    final_state = blog_agentic_ai.invoke(initial_state, config=config)  # type: ignore

    # 🛑 Stop loader
    stop_event.set()
    loader_thread.join()

    print("\n\nFinal Blog Output:\n")
    print(final_state["final_blog"])