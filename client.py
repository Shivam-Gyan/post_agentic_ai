import streamlit as st
import asyncio
from server import blog_agentic_ai
from states import BlogState

st.set_page_config(page_title="Sequential Blog Architect", layout="wide")

# --- UI Header ---
st.title("🚀 Sequential Blog Pipeline")

# --- Layout Definition ---
# Use relative widths to make it responsive
col_process, col_output = st.columns([1, 1.5], gap="medium")

with st.sidebar:
    st.header("Control Panel")
    user_input = st.text_area("Enter Blog Description:", height=150)
    generate_btn = st.button("Start Pipeline", type="primary")

# --- Execution Logic ---
if generate_btn and user_input:
    async def run_pipeline():
        initial_state = BlogState(blog_description=user_input)
        config = {"configurable": {"thread_id": "sequential_run"}}
        
        # This keeps a history of completed node names
        if 'history' not in st.session_state:
            st.session_state.history = []
            
        active_placeholder = col_process.empty() # Single-element container
        final_content = ""

        async for event in blog_agentic_ai.astream(initial_state, config=config):  # type: ignore
            for node_name, state_update in event.items():
                
                # 1. Update History in Column 1
                st.session_state.history.append(node_name)
                
                with active_placeholder.container():
                    # Show previous steps as simple collapsed tags
                    for past_node in st.session_state.history[:-1]:
                        st.write(f"✔️ **{past_node.replace('_', ' ').title()}** (Completed)")
                    
                    # Show CURRENT step expanded
                    with st.expander(f"⚙️ Active: {node_name.replace('_', ' ').title()}", expanded=True):
                        st.json(state_update)

                # 2. Update Column 2 with cumulative content
                if "final_blog" in state_update:
                    final_content = state_update["final_blog"]
                    col_output.markdown(final_content)
                
                if "final_blog" in state_update:
                    final_content = state_update["final_blog"]
                    col_output.markdown("---")
                    col_output.success("Blog Generation Complete!")
                    
                    # Add Download Button at the end of Column 2
                    col_output.download_button(
                        label="💾 Download Final Blog (.md)",
                        data=final_content,
                        file_name="generated_blog.md",
                        mime="text/markdown"
                    )

    asyncio.run(run_pipeline())