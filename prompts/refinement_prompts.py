from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from states import FeedbackStructuredOutputSchema


# prompt to detect intent mode from user query
# def get_intent_detection_prompt(user_query: str) -> list:
#     return [
#         SystemMessage(
#             content=(
#                 "You are an intent classifier. Classify the user query into exactly one mode.\n\n"
#                 "MODES:\n"
#                 "  generate — User wants a NEW blog written. Trigger: any topic/subject the user supplies for content creation.\n"
#                 "  refine   — User wants to EDIT/IMPROVE an EXISTING blog. Trigger: references to changing, shortening, expanding, or restyling current content.\n"
#                 "  chat     — User is greeting, asking questions, or making conversation unrelated to blog creation/editing.\n"
#                 "  publish  — User wants to publish, export, or save the blog. Trigger: words like publish, export, deploy, save, download.\n\n"
#                 "DECISION PRIORITY (apply top-to-bottom, first match wins):\n"
#                 "  1. Contains publish/export/save/deploy intent → publish\n"
#                 "  2. References changing existing blog (e.g. 'make it shorter', 'add examples', 'rewrite intro', 'change tone') → refine\n"
#                 "  3. Supplies a TOPIC or asks to write/create/generate content (e.g. 'blog about X', 'write on Y', 'AI trends', 'semiconductor advances') → generate\n"
#                 "  4. Everything else (greetings, questions, vague/short input with no topic) → chat\n\n"
#                 "FEW-SHOT EXAMPLES:\n"
#                 "  'hi' → chat\n"
#                 "  'hello, how are you?' → chat\n"
#                 "  'what can you do?' → chat\n"
#                 "  'thanks' → chat\n"
#                 "  'write a blog about AI trends in 2026' → generate\n"
#                 "  'semiconductor chip design' → generate\n"
#                 "  'I want a tutorial on LangGraph' → generate\n"
#                 "  'climate change impact on agriculture' → generate\n"
#                 "  'make the tone more formal' → refine\n"
#                 "  'add a code example to section 3' → refine\n"
#                 "  'shorten the conclusion' → refine\n"
#                 "  'publish it' → publish\n"
#                 "  'export as markdown' → publish\n\n"
#                 "Return ONLY the mode value via the function call. No reasoning."
#             )
#         ),
#         HumanMessage(content=user_query),
#     ]



# prompt to generate structured feedback from user refinement query and the current blog
def get_feedback_prompt(user_query: str, final_blog: str, prev_feedback: FeedbackStructuredOutputSchema | None):

    blog_snippet = (
        final_blog if len(final_blog) <= 3000
        else final_blog[:1500] + "\n\n[...truncated...]\n\n" + final_blog[-1500:]
    )

    return [
        SystemMessage(
            content=(
                "You are an Editorial Instruction Generator.\n\n"
                "Your task: Convert the user's request into ONE structured refinement instruction.\n\n"
                "Return structured output with fields:\n"
                "- target_section (exact heading or null if global)\n"
                "- action (single strong verb)\n"
                "- reason (one short sentence)\n"
                "- tone_delta (only if user changes tone)\n"
                "- audience_delta (only if user changes audience)\n"
                "- depth_adjustment (increase, decrease, same)\n"
                "- seo_focus (only if SEO is mentioned)\n\n"
                "Rules:\n"
                "- Do NOT rewrite the blog.\n"
                "- Generate exactly ONE instruction.\n"
                "- If request is vague, choose highest-impact improvement.\n"
                "- Keep reasoning concise.\n"
            )
        ),
        HumanMessage(
            content=(
                f"USER REQUEST:\n{user_query}\n\n"
                f"BLOG:\n{blog_snippet}\n\n"
                f"PREVIOUS FEEDBACK:\n{prev_feedback if prev_feedback else 'None'}"
            )
        ),
    ]




def get_refine_blog_prompt(
    user_query: str,
    blog: str,
    feedback: FeedbackStructuredOutputSchema,
    refined_blog_history: str,
    blog_plan_titles: List[str],
    blog_title: str
) -> list:

    return [
        SystemMessage(
            content=(
                "You are a Senior Editorial Refinement Engine.\n\n"

                "Your task is to apply ONE structured editorial instruction "
                "to an existing blog post.\n\n"

                "This is a controlled refinement — NOT a rewrite.\n\n"

                "CORE RULES:\n"
                "1. Preserve all headings and section order exactly.\n"
                "2. Preserve all hyperlinks, URLs, and citations character-for-character.\n"
                "3. Do not remove existing content unless explicitly instructed.\n"
                "4. Do not invent new facts, statistics, or sources.\n"
                "5. Modify ONLY the minimal text necessary to satisfy the feedback.\n\n"

                "EDITING GUIDELINES:\n"
                "- Improve clarity, depth, flow, or tone as instructed.\n"
                "- Prefer adding or refining sentences over rewriting entire paragraphs.\n"
                "- Do not compress or summarize content.\n"
                "- Leave all unrelated sections completely unchanged.\n\n"

                "OUTPUT:\n"
                "- Return the COMPLETE refined blog in Markdown.\n"
                "- No commentary.\n"
                "- No explanations.\n"
                "- Output only the blog content."
            )
        ),
        HumanMessage(
            content=(
                f"USER REQUEST:\n{user_query}\n\n"
                f"EDITORIAL FEEDBACK:\n{feedback}\n\n"
                f"PREVIOUS REFINED VERSION:\n{refined_blog_history}\n\n"
                f"BLOG TO REFINE:\n{blog}"
                f"BLOG TITLE:\n{blog_title}\n\n"
                f"Title of each Task in the original blog plan:\n{', '.join(blog_plan_titles)}"
            )
        ),
    ]