
import asyncio
import sys
import random
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from nodes import detail_node, fanout, generate_blog_plan, reducer, worker
from states import BlogState
from utils import LOADING_MESSAGES

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
# graph.add_edge('generate_blog_plan', END)
# # 4. initialize checkpointer memory saver
checkpointer = MemorySaver()


# 5. compile the graph
blog_agentic_ai = graph.compile(checkpointer=checkpointer) 

async def async_loading_animation(stop_event: asyncio.Event):
    spinner = ["|", "/", "-", "\\"]
    idx = 0

    while not stop_event.is_set():
        message = random.choice(LOADING_MESSAGES)
        sys.stdout.write(f"\r{spinner[idx % len(spinner)]} {message}")
        sys.stdout.flush()

        idx += 1
        await asyncio.sleep(0.6)

    # Clear line after stop
    sys.stdout.write("\r✅ Blog generation complete!            \n")

async def main():
    print("\n----------- Agentic AI Blog Generator ----------\n")

    initial_state = BlogState(
        blog_description="write a blog on self attention mechanism in deep learning with numerous examples where audience is a beginner in deep learning and has basic understanding of machine learning concepts tone should be practical and crisp with a focus on implementation details and code snippets",
        # blog_topic="Discovery of rocket science and its impact on modern space exploration"
    )

    config = {
        "configurable": {
            "thread_id": "blog_generation_thread",
        }
    }

    stop_event = asyncio.Event()

    # Start the loading animation task
    loader_task = asyncio.create_task(async_loading_animation(stop_event))

    try:
        # 🔥 Async call (graph runs here)
        final_state = await blog_agentic_ai.ainvoke(initial_state, config=config)  # type: ignore

        # 🛑 Stop loader
        stop_event.set()
        await loader_task

        print("\n\nFinal Blog Output:\n")
        print(final_state["final_blog"])

    except Exception as e:
        stop_event.set()
        await loader_task
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")