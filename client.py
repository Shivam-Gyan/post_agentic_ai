import streamlit as st
import asyncio
from server import blog_agentic_ai  # Import your compiled LangGraph app
from states import BlogState # Import your schema

st.set_page_config(page_title="Agentic Blog Architect", layout="wide")

# --- UI Header ---
st.title("🚀 Agentic Blog Architect")
st.markdown("---")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Settings")
    user_input = st.text_area("Enter Blog Description:", 
                              placeholder="e.g., What is self-attention and 2026 updates?")
    run_button = st.button("Generate Blog", type="primary")

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🧠 Agent Thinking & Nodes")
    status_placeholder = st.empty()
    node_data_placeholder = st.container()

with col2:
    st.header("📄 Final Blog Output")
    final_output_placeholder = st.empty()

# --- Execution Logic ---
async def run_agent():
    # Initial state
    initial_state = BlogState(blog_description=user_input)
    config = {"configurable": {"thread_id": "streamlit_run"}}
    
    # We use app.stream to capture node-by-node updates
    async for event in blog_agentic_ai.astream(initial_state, config=config):  # type: ignore
        for node_name, state_update in event.items():
            
            # 1. Update status
            status_placeholder.info(f"Active Node: **{node_name}**")
            
            # 2. Display Node Data in an Expander (Thinking)
            with node_data_placeholder:
                with st.expander(f"Node: {node_name}", expanded=True):
                    st.json(state_update)
            
            # 3. Handle specific output for the UI
            if "final_blog" in state_update and state_update["final_blog"]:
                final_output_placeholder.markdown(state_update["final_blog"])
            
            # Handle sections if they are still being collected
            elif "sections" in state_update:
                with final_output_placeholder.container():
                    st.write("Generating sections...")
                    for section in state_update["sections"]:
                        st.markdown(section)

if run_button and user_input:
    asyncio.run(run_agent())
    st.success("Blog Generation Complete!")