from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from states import PlanSchema, TaskSchema
from langchain_core.messages import SystemMessage, HumanMessage


# prompt to extract details from the initial blog description provided by user
def get_detail_extraction_prompt(blog_description: str) -> list:
    return [
        SystemMessage(
            content=(
                "You are a content analysis and normalization engine.\n"
                "Your task is to extract structured blog metadata from a raw user description.\n\n"

                "STRICT BEHAVIOR RULES:\n"
                "- Do NOT write a blog.\n"
                "- Do NOT add commentary or suggestions.\n"
                "- Do NOT include explanations.\n"
                "- Output must strictly match the expected structured schema.\n\n"

                "EXTRACTION RULES:\n"
                "- topic rules:"
                "  * MUST NOT be empty."
                "  * MUST be a concise, human-readable title."
                "  * 5–12 words."
                "  * If the input lacks a clear title, infer the most appropriate one."
                "- blog_description:"
                "  * Refined version of the input."
                "  * 2–4 concise sentences."
                "  * Preserve original intent, remove noise.\n\n"
                "- audience:\n"
                "  * Be specific (e.g., 'backend developers', 'junior software engineers').\n"
                "  * Do NOT use vague terms like 'everyone' or 'general audience'.\n\n"
                "- tone:\n"
                "  * 1–3 words maximum.\n"
                "  * Examples: practical, technical, instructional, concise, analytical.\n"
                "  * Choose the tone that best matches the description.\n\n"

                "IMPORTANT:\n"
                "- Infer audience and tone ONLY from the input.\n"
                "- If something is unclear, choose the safest, most neutral technical option.\n\n"

                "Output rules:\n"
                "- Output ONLY valid JSON.\n"
                "- Must match the expected structured schema exactly.\n"
                "- No markdown, no comments, no extra text."
            )
        ),
        HumanMessage(
            content=(
                "Raw blog description provided by the user:\n\n"
                f"{blog_description}"
            )
        ),
    ]


# planning Prompt
def get_blog_planning_prompt(
    topic: str,
    description: str,
    audience: str,
    tone: str,
) -> list:
    return [
        SystemMessage(
            content=(
                "You are a senior technical writer and developer advocate.\n"
                "Your task is to produce a precise, actionable plan for a technical blog post.\n\n"

                "GLOBAL CONTEXT (FIXED — DO NOT CHANGE):\n"
                "- Target audience and tone are provided and MUST be respected.\n"
                "- Do NOT invent or modify audience or tone.\n\n"

                "PLANNING INTENT (IMPORTANT):\n"
                "- Use the blog topic and description to decide WHICH sections are needed.\n"
                "- Adapt section focus, depth, and ordering based on:\n"
                "  * topic complexity\n"
                "  * description emphasis\n"
                "  * audience seniority\n"
                "- Do NOT use a generic or templated outline.\n\n"

                "HARD REQUIREMENTS (MUST FOLLOW):\n"
                "- Create 5–7 sections total.\n"
                "- Each section MUST conform exactly to TaskSchema.\n"
                "- Include EXACTLY ONE section with section_type = 'common_mistakes'.\n"
                "- All section IDs must be sequential integers starting from 1.\n\n"

                "TaskSchema requirements (per section):\n"
                "- title: short, clear, technical section heading.\n"
                "- goal: exactly ONE sentence describing what the reader will be able to do or understand.\n"
                "- bullets: 3–5 concrete, non-overlapping, actionable subpoints.\n"
                "- target_words: integer between 120 and 450.\n"
                "- section_type: one of {intro, core, examples, checklist, common_mistakes, conclusion}.\n\n"

                "Technical quality bar:\n"
                "- Use correct engineering terminology appropriate for the given audience.\n"
                "- Prefer engineering flow: problem → intuition → approach → implementation → trade-offs → validation.\n"
                "- Bullets must be actionable and testable (build, measure, compare, verify).\n"
                "- Avoid vague bullets like 'Explain X' or 'Discuss Y'.\n\n"

                "Coverage requirements (across the entire plan, include AT LEAST ONE):\n"
                "- Minimal working example or code sketch.\n"
                "- Edge cases or failure modes.\n"
                "- Performance or cost considerations.\n"
                "- Security or privacy considerations (if relevant).\n"
                "- Debugging, observability, or testing guidance.\n\n"

                "Ordering guidance:\n"
                "- Start with an intro/problem framing section.\n"
                "- Build core concepts before advanced details.\n"
                "- Place 'common_mistakes' where it naturally fits.\n"
                "- End with a conclusion or checklist section.\n\n"

                "Output rules:\n"
                "- Output ONLY valid JSON.\n"
                "- Must match PlanSchema EXACTLY.\n"
                "- Do NOT include markdown, comments, or explanations."
            )
        ),
        HumanMessage(
            content=(
                f"Blog topic:\n{topic}\n\n"
                f"Blog description:\n{description}\n\n"
                f"Target audience:\n{audience}\n\n"
                f"Writing tone:\n{tone}"
            )
        ),
    ]


# worker prompt to generate each section of the blog as per task in plan
def worker_prompt(task: TaskSchema, blog_topic: str, plan: PlanSchema, audience: str, tone: str) -> list:
    bullets_text = "\n".join(f"- {b}" for b in task.bullets)

    return [
        SystemMessage(
            content=(
                "You are a senior technical writer and developer advocate.\n"
                "Write EXACTLY ONE section of a technical blog post in Markdown.\n\n"

                "HARD CONSTRAINTS:\n"
                "- Follow the provided Goal.\n"
                "- Cover ALL bullets in the given order (do NOT skip, merge, or reorder).\n"
                "- Stay within ±15% of the target word count.\n"
                "- Output ONLY the section content in Markdown.\n"
                "- No questions, no commentary, no summaries outside the section.\n\n"

                "Technical quality bar:\n"
                "- Be implementation-oriented and precise.\n"
                "- Prefer concrete details: APIs, data structures, protocols, algorithms.\n"
                "- When relevant, include at least ONE of:\n"
                "  * minimal code snippet (correct and idiomatic)\n"
                "  * example input/output\n"
                "  * checklist\n"
                "  * text-described diagram (e.g., Flow: A → B → C)\n"
                "- Mention trade-offs briefly (performance, cost, complexity, reliability).\n"
                "- Call out edge cases or failure modes explicitly.\n"
                "- If a best practice is mentioned, include the WHY in one sentence.\n\n"

                "Markdown style rules:\n"
                "- Start with: ## <Section Title>\n"
                "- Use short paragraphs and bullet lists where helpful.\n"
                "- Use fenced code blocks for code.\n"
                "- Avoid fluff and marketing language."
            )
        ),
        HumanMessage(
            content=(
                f"Blog title: {plan.blog_title}\n"
                f"Audience: {audience}\n"
                f"Tone: {tone}\n"
                f"Topic: {blog_topic}\n\n"

                f"Section title: {task.title}\n"
                f"Section type: {task.section_type}\n"
                f"Goal: {task.goal}\n"
                f"Target words: {task.target_words}\n"
                f"Bullets:\n{bullets_text}"
            )
        ),
    ]

