
from typing import cast
from langgraph.graph import END
from langchain_core.messages import HumanMessage, RemoveMessage, AIMessage
from prompts.conversation_prompt import get_conversation_summary_prompt
from states import BlogState, SummaryStructuredOutputSchema
from utils import parse_mode
from models import conversation_summary_structured_output_model


async def intent_detection_node(state: BlogState):

    try:
        mode, cleaned_query = parse_mode(state.user_query)

        if not mode and not cleaned_query:
            raise ValueError("Intent detection failed to parse mode and query")

        # If user requested a refinement but we don't have a generated blog yet,
        # return an AI suggestion message and short-circuit the node.
        if not mode or (mode == 'refine' and not getattr(state, 'final_blog', None)):
            suggestion = (
                "To refine the content, please generate the initial blog draft first. "
                "I can then provide targeted refinements, polishing, and structural improvements.\n"
                "Comment: No existing draft blog was found in the current state."
            )

            if not mode:
                suggestion = (
                    "I couldn't detect a clear intent in your query. "
                    "Please start your query with 'generate:', 'refine:', 'chat:', or 'publish:' to indicate your desired action.\n"
                    "Comment: No mode prefix detected in the user query."
                )

            ai_msg = AIMessage(content=suggestion)

            return {
                "mode": "guard",
                "messages": [ai_msg],
                "summary": state.summary,
            }

        # summarization of message history 

        updated_summary = None
        messages_to_remove = None


        keep_last = 3  # number of recent messages to keep after trimming

        # checking the size of the conversation history messages to reduce and summarize
        if len(state.messages) > 6:

            print(f"\n\nOriginal Messages before summary : \n{state.messages}")
            # summarize all messages EXCEPT the most recent ones
            messages_to_summarize = state.messages[:-keep_last]

            prompt = get_conversation_summary_prompt(
                messages_to_summarize,
                previous_summary=cast(SummaryStructuredOutputSchema, state.summary)
            )

            try:
                updated_summary = await conversation_summary_structured_output_model.ainvoke(prompt)
                print(f"\n\nMessages Summary generated after LLM call: \n {updated_summary}")
                print("\n\n")
            except Exception as summary_err:
                print(f"Warning: summary model call failed ({summary_err}); keeping existing summary.")
                updated_summary = None

            # mark old messages for removal using RemoveMessage
            messages_to_remove = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]
        else:
            messages_to_remove = None

        user_message = HumanMessage(content=cleaned_query)

        result: dict = {
            'user_query': cleaned_query,
            'mode': mode,
            "messages": [user_message],
            "summary": updated_summary if updated_summary else state.summary,
        }

        # if we trimmed, also send remove ops so add_messages actually drops the old ones
        if messages_to_remove:
            result["messages"] = messages_to_remove + [user_message]


        # Append the current user message to conversation history.
        # operator.add on conversation_state.messages will merge this
        # with the existing checkpoint, so history accumulates across turns.
        return result  # type: ignore

    except Exception as e:
        print(f"Error in intent_detection_node: {e}")
        # Preserve summary and add user message so the conversation isn't lost.
        # Default to 'chat' mode so the graph doesn't crash on None mode.
        safe_mode = getattr(state, 'mode', None) or 'chat'
        fallback_msg = HumanMessage(content=state.user_query or "")
        return {
            'mode': safe_mode,
            'summary': state.summary,
            'messages': [fallback_msg],
        }


def intent_router_func(state:BlogState) -> str:
    mode = state.mode

    if mode == "guard":
        return END
    elif mode == "generate":
        return 'router_node'
    elif mode == "refine":
        return 'refine_subgraph'
    elif mode == "chat":
        return 'conversation_subgraph'
    elif mode == "publish":
        return 'publish_node'
    else:
        # default to router_node for generation if mode is unrecognized
        return 'conversation_subgraph'


