
from typing import cast
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import RemoveMessage
from models import get_generation_model
from prompts.conversation_prompt import get_conversation_prompt
from states import BlogState, SummaryStructuredOutputSchema


async def chat_node_func(state: BlogState):

    try:
        #  fetch the conversation summary and history from state
        summary = state.summary
        history = state.messages[-20:]

        # print(f"\n\nConversation History passed to chat node: \n {history}")
        # print(f"\n\nConversation Summary passed to chat node: \n {summary}")


        # print(f"\n\nOriginal Messages before summary : \n{state.messages}")

        if not history:
            history = [HumanMessage(content=state.user_query)]

        # generate the conversation prompt with the history and summary
        conversation_prompt = get_conversation_prompt(
            messages=history,
            summary=summary if summary else state.summary
        )



        # Import `agent` here to avoid circular imports at module import time.
        import agent

        gen = get_generation_model()
        agent_tools = getattr(agent, "tools", None)
        if agent_tools:
            gen = gen.bind_tools(agent_tools)

        response = await gen.ainvoke(conversation_prompt)

        return {
            "messages": [response]
        }

    except Exception as e:
        print(f"Error in chat_node_func: {e}")
        raise e