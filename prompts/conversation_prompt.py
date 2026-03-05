import json
from typing import Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from states import SummaryStructuredOutputSchema




def get_conversation_summary_prompt(
    oldest_messages: Sequence[BaseMessage],
    previous_summary: SummaryStructuredOutputSchema
) -> str:

    prompt = """
### ROLE
You are a Memory Manager for a 3-branch Blog Agent (Generate, Refine, Chat). Your core job is to produce a **Cumulative Update** of the user's state ($S_{new} = S_{old} + \\text{New Segment}$). You must merge historical context with fresh information seamlessly.

### STRATEGIC RULES
1. **Cumulative Merging ($S_{new} = S_{old} + \\text{New Segment}$)**: 
   - Start with the 'Existing Memory'. 
   - Update fields ONLY if the 'New Segment' provides new, specific info. 
   - **CRITICAL**: NEVER return `null` or empty lists for fields that already have data in 'Existing Memory' unless the user explicitly retracted that info.
2. **Goal Activation (`user_goal`)**: 
   - This field is the "North Star" project mission. 
   - If the user is just asking questions or exploring (Chat branch), keep this `null` (or strictly preserve its existing value). 
   - The moment the user gives a command to build something (e.g., "Write a blog about Agentic AI", "Draft a tutorial"), update this field to reflect the active task.
3. **Bio Persistence**: Professional roles (e.g., "I am an AI Engineer") are permanent traits. Map them to `user_professional_bio` and carry them forward forever.
4. **List Deduplication**: When adding to lists (`current_topics_of_interest`, `constraints`, `decisions_made`), append new items without duplicating existing ones.

### FIELD DEFINITIONS
- **user_real_name**: The user's actual name.
- **user_professional_bio**: Their role/expertise. Use this to gauge technical depth.
- **current_topics_of_interest**: A running list of subjects explored during the 'Chat' branch (e.g., ML mistakes, Black holes).
- **user_goal**: The active deliverable or task (e.g., "Drafting a post on LLM Bias"). See Rule 2.
- **audience**: Target readers for the blog.
- **decisions_made**: Confirmed technical or stylistic choices.
- **constraints**: Hard limitations or requirements for the blog (e.g., "No code snippets", "Under 500 words").
- **preferences**: User's soft stylistic or content preferences (e.g., "Witty tone").
- **open_questions**: Pending clarifications required from the user. ONLY include questions the AI explicitly asked the user that the user HAS NOT YET answered. Do NOT include questions the user asked.
--------------------------------
Existing Memory (JSON)
--------------------------------
"""
    prompt += json.dumps(previous_summary.model_dump(), indent=2)

    prompt += "\n\n--------------------------------\nNew Conversation Segment\n--------------------------------\n"
    for msg in oldest_messages:
        role = "User" if msg.type == "human" else "Assistant"
        prompt += f"{role}: {msg.content[:500]}\n"

    prompt += """
--------------------------------
FINAL INSTRUCTION:
Synthesize and output the updated JSON memory. 
- Merge the New Conversation Segment into the Existing Memory.
- DO NOT WIPE PREVIOUSLY STORED DATA. If a field in Existing Memory has a value, keep it unless it is directly superseded by the New Segment.
- Set or update the `user_goal` ONLY if a clear request to generate or refine content was made.
"""
    return prompt

def get_conversation_prompt(
    messages: Sequence[BaseMessage],
    summary: SummaryStructuredOutputSchema
) -> str:

    prompt = """
### ROLE
You are a highly capable AI assistant engaging in conversation. You leverage long-term memory to maintain a seamless, personalized, and context-aware dialogue.

### RESPONSE RULES
1. **Answer the Latest Message:** Focus entirely on the user's most recent input.
2. **Adapt to Expertise:** Always match your technical depth to the user's `user_professional_bio`. If they are an expert (e.g., AI Engineer), do not give beginner-level explanations.
3. **Respect Constraints & Preferences:** If the memory contains format rules (e.g., "short answer", "no emojis") or style preferences, follow them exactly.
4. **Contextual Continuity:** Use `user_goal` ONLY if the user's latest message explicitly relates to an active project. **Never** reference previous topics or past conversation subjects unless the user's latest message directly mentions them.
5. **No Meta-Talk:** Do NOT mention your memory, JSON, prompts, or reasoning. Just speak directly to the user.
6. **Be Concise:** Do not add lengthy explanations or fluff unless explicitly requested.
7. **Topic Isolation:** If the user's latest message is on a NEW subject (e.g., a greeting, a new question), treat it as a standalone query. Do NOT connect it to prior topics (e.g., books discussed earlier) unless the user explicitly asks.

--------------------------------
STRUCTURED MEMORY (Long-term Context)
--------------------------------
"""
    # Filter memory to ONLY include fields relevant to casual/technical chat.
    # current_topics_of_interest is intentionally excluded: sending it causes the
    # LLM to force-connect every reply to previous topics (e.g., The Alchemist)
    # even when the user asks something completely unrelated.
    chat_relevant_keys = [
        "user_real_name", 
        "user_professional_bio", 
        "current_topics_of_interest", 
        "preferences", 
        "constraints", 
        "user_goal" # Kept so the chat knows if you are taking a break from an active project
    ]

    # Build active memory: Only include keys that are relevant AND not empty
    active_memory = {
        k: v for k, v in summary.model_dump().items() 
        if v and k in chat_relevant_keys
    }

    prompt += json.dumps(active_memory, indent=2)

    prompt += "\n\n--------------------------------\n"
    prompt += "RECENT CONVERSATION (Short-term Context)\n"
    prompt += "--------------------------------\n"

    # Separate history from the latest message so the LLM cannot miss it
    prior_messages = list(messages[:-1]) if len(messages) > 1 else []
    latest_message = messages[-1] if messages else None

    for msg in prior_messages:
        role = "User" if msg.type == "human" else "Assistant"
        prompt += f"{role}: {msg.content[:500]}\n"

    prompt += "\n--------------------------------\n"
    prompt += "LATEST USER MESSAGE — YOU MUST RESPOND TO THIS AND ONLY THIS\n"
    prompt += "--------------------------------\n"
    if latest_message:
        prompt += f"{latest_message.content[:500]}\n"

    prompt += """
--------------------------------
FINAL TASK
--------------------------------
Respond ONLY to the LATEST USER MESSAGE above.
Ignore previous topics unless the latest message explicitly references them.
Return only the final answer. Do not explain your reasoning.
"""

    return prompt