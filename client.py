import streamlit as st
import asyncio
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from states import BlogState        # Your Pydantic State


# Cache the compiled graph so MemorySaver survives Streamlit reruns / file-watcher reloads
@st.cache_resource
def get_blog_app():
    from server import build_blog_graph
    return build_blog_graph()


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback

def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        # LangGraph updates usually come as {node_name: {state_updates}}
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
st.set_page_config(page_title="Blog Generation Agentic AI Dashboard", layout="wide")

st.title("Blog Generation Agentic AI Dashboard")

# Initialize Session State
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "interrupted" not in st.session_state:
    st.session_state["interrupted"] = False
if "pipeline_config" not in st.session_state:
    st.session_state["pipeline_config"] = None

# --- Sidebar Control Panel ---
with st.sidebar:
    st.header("Control Panel")
    topic_input = st.text_area("Enter Detailed Blog Description:", height=150)
    as_of = st.date_input("As-of date", value=date.today())
    generate_btn = st.button("Start Generation", type="primary")
    
    st.divider()
    if st.button("Clear History"):
        st.session_state["last_out"] = None
        st.session_state["logs"] = []
        st.session_state["interrupted"] = False
        st.session_state["pipeline_config"] = None
        get_blog_app.clear()  # Reset MemorySaver so next run starts fresh
        st.rerun()

    # --- Publish to Feather Feable (HITL) ---
    if st.session_state["interrupted"]:
        st.divider()
        st.header("📤 Publish Blog")
        st.write("Blog generated! Publish to **Feather Feable**?")

        publish_choice = st.radio(
            "Publish to Feather Feable?",
            ["Yes", "No"],
            index=1,
            key="publish_choice",
        )

        if publish_choice == "Yes":
            access_token = st.text_input(
                "Enter your Feather Feable access token:",
                type="password",
                key="ff_access_token",
            )
            if st.button("🚀 Publish Now", type="primary"):
                if access_token.strip():
                    async def resume_publish():
                        cfg = st.session_state["pipeline_config"]
                        status = st.status("Publishing to Feather Feable...", expanded=True)
                        async for event in get_blog_app().astream(
                            Command(resume={"approved": True, "access_token": access_token.strip()}),
                            config=cfg,
                            stream_mode="updates",
                        ):
                            for node_name, state_update in event.items():
                                status.write(f"✔️ Node: `{node_name}`")
                                log_entry = state_update
                                if hasattr(state_update, "model_dump"):
                                    log_entry = state_update.model_dump()
                                st.session_state["logs"].append(
                                    f"[{node_name}] {json.dumps(log_entry, default=str)[:500]}..."
                                )
                                if isinstance(state_update, dict) and st.session_state["last_out"]:
                                    st.session_state["last_out"].update(state_update)
                        st.session_state["interrupted"] = False
                        status.update(label="✅ Done!", state="complete", expanded=False)

                    asyncio.run(resume_publish())
                    st.rerun()
                else:
                    st.warning("Please enter your access token.")
        else:
            if st.button("⏭️ Skip & Finish"):
                async def resume_skip():
                    cfg = st.session_state["pipeline_config"]
                    async for event in get_blog_app().astream(
                        Command(resume={"approved": False}),
                        config=cfg,
                        stream_mode="updates",
                    ):
                        for node_name, state_update in event.items():
                            if isinstance(state_update, dict) and st.session_state["last_out"]:
                                st.session_state["last_out"].update(state_update)
                    st.session_state["interrupted"] = False

                asyncio.run(resume_skip())
                st.rerun()


#  adding this CSS to increase the font size of the tabs and add spacing between them for better readability and aesthetics.
st.markdown("""
    <style>
        /* Target the tab bar container */
        [data-testid="stTabs"] {
            gap: 30px; /* Space between tabs */
        }

        /* Target each individual tab button */
        button[data-baseweb="tab"] {
            font-size: 20px; /* Increase font size */
            font-weight: 600;
            padding: 10px 20px; /* Increase padding for better click area */
        }

        /* Target the text inside the tab */
        button[data-baseweb="tab"] p {
            font-size: 20px; 
        }
    </style>
""", unsafe_allow_html=True)

# --- Main Tabs ---
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🧾 Logs"]
)

# --- Execution Logic ---
# --- Execution Logic (Updated with live Queries in JSON) ---
if generate_btn:
    if not topic_input.strip():
        st.warning("Please enter a description.")
        st.stop()

    async def run_pipeline():
        initial_state = BlogState(blog_description=topic_input)
        config = {"configurable": {"thread_id": "blog_agentic_ai_1"}}
        
        status = st.status("Running Agent Pipeline...", expanded=True)
        progress_area = st.empty()
        
        current_state_dict: Dict[str, Any] = {}
        
        try:
            async for event in get_blog_app().astream(initial_state, config=config, stream_mode="updates"):#type: ignore
                for node_name, state_update in event.items():
                    status.write(f"✔️ Node: `{node_name}`")
                    
                    # Merge updates
                    current_state_dict = extract_latest_state(current_state_dict, state_update)
                    
                    # Create the live summary for the JSON display
                    summary = {
                        "current_node": node_name,
                        "research_mode": current_state_dict.get("research_mode"),
                        "research_queries": current_state_dict.get("research_queries", []), # Added this
                        "evidence_found": len(current_state_dict.get("evidence", [])),
                        "sections_generated": len(current_state_dict.get("sections", [])),
                    }
                    
                    # Display the JSON
                    progress_area.json(summary)
                    
                    # Log the raw update (using model_dump if it's a pydantic object)
                    log_entry = state_update
                    if hasattr(state_update, "model_dump"):
                        log_entry = state_update.model_dump()
                    
                    st.session_state["logs"].append(f"[{node_name}] {json.dumps(log_entry, default=str)[:500]}...")
        except GraphInterrupt:
            # Expected — graph paused at interrupt() in publish_node
            pass
        except Exception as exc:
            # Catch wrapped GraphInterrupt (may be raised as tuple in some LangGraph versions)
            if "Interrupt" in type(exc).__name__ or "interrupt" in str(exc).lower():
                pass  # Expected HITL interrupt
            else:
                raise

        st.session_state["last_out"] = current_state_dict

        # Check if graph is paused at an interrupt (HITL)
        graph_state = await get_blog_app().aget_state(config)  # type: ignore[arg-type]
        if graph_state.next:  # graph is interrupted, not finished
            st.session_state["interrupted"] = True
            st.session_state["pipeline_config"] = config
            status.update(label="⏸️ Blog ready — check sidebar to publish!", state="running", expanded=False)
        else:
            st.session_state["interrupted"] = False
            status.update(label="✅ Generation Complete!", state="complete", expanded=False)

    asyncio.run(run_pipeline())
    # Rerun so the sidebar HITL section (rendered earlier) picks up the interrupted flag
    if st.session_state.get("interrupted"):
        st.rerun()

# --- Render Results ---
out = st.session_state.get("last_out")

if out:
    # 1. Plan Tab
    # --- Replace the "Plan Tab" section in the previous code with this ---

# --- Plan Tab Updated for Pydantic V2 and Streamlit 2026 standards ---
    with tab_plan:
        st.subheader("Content Strategy & Research Plan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Research Mode", out.get("research_mode", "N/A").replace("_", " ").title())
        with col2:
            st.metric("Blog Kind", out.get("blog_kind", "N/A").title())
        with col3:
            st.metric("Audience", out.get("audience", "N/A"))

        queries = out.get("research_queries", [])
        if queries:
            with st.expander("🔎 Generated Research Queries", expanded=True):
                for q in queries:
                    st.write(f"- {q}")

        st.divider()

        plan = out.get("plan")
        if plan:
            # Pydantic V2 use model_dump()
            plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
            st.write(f"**Draft Title:** {plan_dict.get('blog_title', 'Untitled Blog')}")
            
            tasks = plan_dict.get("tasks", [])
            if tasks:
                formatted_tasks = []
                for t in tasks:
                    t_data = t.model_dump() if hasattr(t, "model_dump") else t
                    formatted_tasks.append({
                        "ID": t_data.get("id"),
                        "Section": t_data.get("title"),
                        "Goal": t_data.get("goal"),
                        "Words": t_data.get("target_words"),
                        "Type": t_data.get("section_type"),
                        "Research?": "✅" if t_data.get("require_research") else "❌"
                    })
                # Updated width parameter for Streamlit 1.x (2026)
                st.dataframe(pd.DataFrame(formatted_tasks), width="stretch", hide_index=True)
                
                with st.expander("View Full Task Details"):
                    st.json(tasks)

    # --- Evidence Tab Updated ---
    with tab_evidence:
        st.subheader("Research Evidence")
        evidence_list = out.get("evidence", [])
        if evidence_list:
            rows = []
            for e in evidence_list:
                e_data = e.model_dump() if hasattr(e, "model_dump") else e
                rows.append({
                    "Title": e_data.get("title"),
                    "URL": e_data.get("url"),
                    "Content Preview": (e_data.get("content") or "")[:100] + "..."
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No external evidence gathered.")


    # 3. Preview Tab
    with tab_preview:
        st.subheader("Final Blog Output")
        final_md = out.get("final_blog", "")
        if final_md:
            st.markdown(final_md)
            
            # Download Button
            title = out.get("blog_title") or extract_title_from_md(final_md, "generated_blog")
            st.download_button(
                label="💾 Download Markdown (.md)",
                data=final_md,
                file_name=f"{safe_slug(title)}.md",
                mime="text/markdown"
            )
        else:
            st.warning("The blog body is currently empty.")

        # --- Publish Result ---
        publish_result = out.get("publish_result", "")
        if publish_result:
            if "✅" in publish_result:
                st.success(publish_result)
            elif "❌" in publish_result:
                st.error(publish_result)
            else:
                st.info(publish_result)

    # 4. Logs Tab
    with tab_logs:
        st.subheader("Execution History")
        log_text = "\n\n".join(st.session_state["logs"])
        st.text_area("Trace", value=log_text, height=400)

else:
    with tab_preview:
        st.info("Enter a description in the sidebar and click 'Start Generation' to begin.")