import asyncio
from node import intent_detection_nodes
from states import BlogState, SummaryStructuredOutputSchema
from langchain_core.messages import HumanMessage, RemoveMessage


async def run_test():
    # create a long conversation history to trigger summarization
    msgs = []
    for i in range(1, 25):
        m = HumanMessage(content=f"message {i}")
        # give an id attribute so RemoveMessage can reference it
        setattr(m, "id", f"m{i}")
        msgs.append(m)

    state = BlogState(messages=msgs, user_query="chat: hello")

    # Patch the model used inside intent_detection_nodes with a dummy that returns a dict
    class DummyModel:
        async def ainvoke(self, prompt):
            return {
                "user_real_name": None,
                "user_professional_bio": "AI Engineer",
                "current_topics_of_interest": ["ML", "Black Holes"],
                "user_goal": "Write a blog",
                "audience": "Beginner developers",
                "constraints": [],
                "preferences": [],
                "decisions_made": [],
                "open_questions": []
            }

    # Replace the model reference in the node module (import was at module scope)
    intent_detection_nodes.conversation_summary_structured_output_model = DummyModel()

    result = await intent_detection_nodes.intent_detection_node(state)

    summary = result.get("summary")
    print("Returned summary type:", type(summary))
    print(summary)

    # Basic checks
    if isinstance(summary, SummaryStructuredOutputSchema):
        print("PASS: summary is validated and is SummaryStructuredOutputSchema")
    else:
        print("FAIL: summary is NOT a SummaryStructuredOutputSchema instance")

    messages = result.get("messages")
    if messages and isinstance(messages[0], RemoveMessage):
        print("PASS: messages include RemoveMessage (old messages marked for removal)")
    else:
        print("WARN: messages do not include RemoveMessage; trimming may not have occurred")


if __name__ == "__main__":
    asyncio.run(run_test())
