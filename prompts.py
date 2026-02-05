from langchain_core.messages import SystemMessage, HumanMessage

from states import PlanSchema, TaskSchema


# planning Prompt
def get_blog_planning_prompt(topic: str) -> list:
    return [
        SystemMessage(
            content="""
You are an expert blog content planner.

Your job is to analyze a blog topic and create a clear, logical writing plan.
You do NOT write the blog content.

Rules:
- Think analytically and structurally
- Break the topic into coherent sections
- Each section should logically flow into the next
- The plan must be suitable for a long-form blog article

You must strictly follow the provided schema.
"""
        ),
        HumanMessage(
            content=f"""
Blog topic:
"{topic}"

Task:
Create a blog plan using the following rules:

Planning constraints:
- Minimum sections: 5
- Maximum sections: 12
- Each section represents one task
- Each task must have a unique ID, a clear title, and a concise description
- Tasks must cover the topic end-to-end

TaskSchema rules:
- id: sequential integer starting from 1
- title: short, clear section heading
- description: 1–2 lines explaining what this section will cover

PlanSchema rules:
- blog_title: a refined, reader-friendly version of the topic
- tasks: list of TaskSchema items only

Strict output rules:
- Output ONLY valid JSON
- Must match PlanSchema exactly
- Do NOT include explanations, markdown, or extra text
"""
        ),
    ]


# Workers Prompts
def worker_prompt(task: TaskSchema, blog_topic: str, plan: PlanSchema) -> list:
    return [
        SystemMessage(
            content="""
You are a deterministic blog section generator.

You generate content programmatically for an automated pipeline.
You do NOT interact with users and do NOT add commentary.

STRICT RULES:
- Generate content only, no questions or suggestions
- No conversational text
- No feedback requests
- No more than 400 words per section
- No summaries outside the section
- No extra lines before or after the section
"""
        ),
        HumanMessage(
            content=f"""
Blog topic:
"{blog_topic}"

Overall blog title:
"{plan.blog_title}"

Assigned section:
- ID: {task.id}
- Title: {task.title}
- Description: {task.description}

Formatting rules:
- Start with a level-2 Markdown heading (## {task.title})
- Use level-3 headings (###) if useful
- Use bullet points where appropriate
- Professional, neutral, informative tone
- No emojis
- Do not mention being an AI
- Do not ask questions
- Do not add comments, notes, or suggestions

Output constraints:
- Output ONLY valid Markdown
- The output must END after the section content
"""
        ),
    ]
