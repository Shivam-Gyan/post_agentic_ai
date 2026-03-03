



from states import BlogState
from utils import parse_mode,blog


async def intent_detection_node(state: BlogState) :

    try:

        # response = await intent_structured_output_model.ainvoke(get_intent_detection_prompt(state.user_query)) #type: ignore
        # print(f"Intent detection response: {response}")

        mode, cleaned_query = parse_mode(state.user_query)

        if not mode and not cleaned_query:
            raise ValueError("Intent detection failed to parse mode and query")
     
        return {'user_query':cleaned_query, 'mode': mode,"final_blog":blog} #type: ignore
    
    except Exception as e:
        print(f"Error in intent_detection_node: {e}")
        # default to generate mode if intent detection fails
        return {'mode': 'generate'}


def intent_router_func(state:BlogState) -> str:
    mode = state.mode

    if mode == "generate":
        return 'router_node'
    elif mode == "refine":
        return 'refine_subgraph'
    elif mode == "chat":
        return 'chat_node'
    elif mode == "publish":
        return 'publish_node'
    else:
        # default to router_node for generation if mode is unrecognized
        return 'router_node'


