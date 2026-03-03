
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage
from models import get_generation_model

from states import BlogState, ConversationState



async def chat_node_func(state: BlogState):

    try:
        # conversation_state.messages already contains the new HumanMessage
        # (merged via operator.add from the input passed in server.py)
        # Take last 5 messages for context window
        history = state.messages[-5:]
        print(f"Chat node history: {history}")

        # fall back to the raw query if history is somehow empty
        messages_to_send = history if history else [HumanMessage(content=state.user_query)]

        response = await get_generation_model().ainvoke(messages_to_send)
        # print(f"\n\nChat node response: {response}")
        return {
            "messages": [AIMessage(content=response.content)]
        }
    
    except Exception as e:
        print(f"Error in chat_node_func: {e}")
        raise e

    
