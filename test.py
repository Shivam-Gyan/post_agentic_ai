"""
LangGraph Blog Agent
====================
Architecture:
  START → Intent → BlogGenerator (once) → END
                 → RefineSubgraph       → END
                 → ChatNode             → END
  (loop lives in the Python while loop, not inside the graph)
"""

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    mode: str                    # "generate" | "refine" | "chat"
    messages: list
    final_blog: Optional[str]


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=GROQ_API_KEY,  # type: ignore
    temperature=0.4,
)


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def intent_node(state: AgentState) -> AgentState:
    """Sets mode based on user message and whether a blog exists."""
    if not state.get("final_blog"):
        return {**state, "mode": "generate"}

    last_msg = _last_user_message(state).lower()

    if any(k in last_msg for k in ["refine", "improve", "change", "edit", "update", "rewrite"]):
        mode = "refine"
    elif any(k in last_msg for k in ["new blog", "create another", "generate new", "start over"]):
        mode = "generate"
    else:
        mode = "chat"

    return {**state, "mode": mode}


def blog_generator_node(state: AgentState) -> AgentState:
    """Generates a blog post from the user's request."""
    last_msg = _last_user_message(state)

    response = llm.invoke([
        SystemMessage(content=(
            "You are a professional blog writer. "
            "Write a well-structured, engaging blog post based on the user's request. "
            "Include a title, introduction, 2-3 sections, and conclusion."
        )),
        HumanMessage(content=last_msg),
    ])

    blog = response.content
    ai_reply = (
        f"✅ Blog generated!\n\n{blog}\n\n"
        "---\nYou can now:\n"
        "• Ask me to **refine** any part\n"
        "• Ask a **question** about the blog\n"
        "• Say **'new blog'** to start fresh"
    )

    return {
        **state,
        "final_blog": blog,
        "mode": "chat",
        "messages": state["messages"] + [AIMessage(content=ai_reply)],
    }


def chat_node(state: AgentState) -> AgentState:
    """Handles general conversation about the blog."""
    last_msg = _last_user_message(state)
    context = f"Current blog:\n{state.get('final_blog', 'No blog yet.')}"

    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful assistant. The user has generated a blog post. "
            "Answer their questions or comments about it naturally.\n\n" + context
        )),
        HumanMessage(content=last_msg),
    ])

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


# ─────────────────────────────────────────────
# REFINE SUBGRAPH
# ─────────────────────────────────────────────

def refine_node(state: AgentState) -> AgentState:
    """Refines the existing blog per the user's instruction."""
    last_msg = _last_user_message(state)
    current_blog = state.get("final_blog", "")

    response = llm.invoke([
        SystemMessage(content=(
            "You are a professional editor. The user wants to refine their blog post. "
            "Apply their requested changes and return the complete updated blog post only."
        )),
        HumanMessage(content=(
            f"Current blog:\n{current_blog}\n\n"
            f"User's refinement request: {last_msg}"
        )),
    ])

    refined_blog = response.content
    ai_reply = (
        f"✏️ Blog refined!\n\n{refined_blog}\n\n"
        "---\nWant more refinements or have questions?"
    )

    return {
        **state,
        "final_blog": refined_blog,
        "mode": "chat",
        "messages": state["messages"] + [AIMessage(content=ai_reply)],
    }


def build_refine_subgraph():
    subgraph = StateGraph(AgentState)
    subgraph.add_node("refine_node", refine_node)
    subgraph.add_edge(START, "refine_node")
    subgraph.add_edge("refine_node", END)
    return subgraph.compile()


# ─────────────────────────────────────────────
# ROUTER (edge function)
# ─────────────────────────────────────────────

def route_after_intent(state: AgentState) -> Literal["blog_generator", "refine_subgraph", "chat_node"]:
    """Single routing decision after intent_node sets the mode."""
    mode = state.get("mode", "chat")
    if mode == "generate":
        return "blog_generator"
    elif mode == "refine":
        return "refine_subgraph"
    else:
        return "chat_node"


# ─────────────────────────────────────────────
# MAIN GRAPH  ← the fix is here
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    refine_subgraph = build_refine_subgraph()

    graph.add_node("intent", intent_node)
    graph.add_node("blog_generator", blog_generator_node)
    graph.add_node("refine_subgraph", refine_subgraph)
    graph.add_node("chat_node", chat_node)

    graph.add_edge(START, "intent")

    # ONE conditional edge from intent → all three destinations
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "blog_generator":  "blog_generator",
            "refine_subgraph": "refine_subgraph",
            "chat_node":       "chat_node",
        },
    )

    graph.add_edge("blog_generator",  END)
    graph.add_edge("refine_subgraph", END)
    graph.add_edge("chat_node",       END)

    return graph.compile()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    app = build_graph()

    state: AgentState = {
        "mode": "generate",
        "messages": [],
        "final_blog": None,
    }

    print("🤖 Blog Agent ready. Tell me what blog to write!\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = app.invoke(state)

        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\nAssistant: {msg.content}\n")
                break


if __name__ == "__main__":
    main()