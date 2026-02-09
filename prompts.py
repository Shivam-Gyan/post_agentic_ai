from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from states import EvidencePackSchema, EvidenceSchema, PlanSchema, TaskSchema
from langchain_core.messages import SystemMessage, HumanMessage


# prompt to extract details from the initial blog description provided by user
def get_router_prompt(blog_description: str) -> list:
    return [
        SystemMessage(
            content=(
                "You are a content analysis and normalization engine.\n"
                "Your task is to extract structured blog metadata from a raw user description "
                "and populate all fields required by the ResearchSchema.\n\n"

                "STRICT BEHAVIOR RULES:\n"
                "- Do NOT write a blog.\n"
                "- Do NOT add commentary or suggestions.\n"
                "- Do NOT include explanations.\n"
                "- Ensure the output strictly conforms to the structured schema.\n\n"

                "EXTRACTION RULES:\n"

                "topic:\n"
                "- MUST NOT be empty.\n"
                "- MUST be a concise, human-readable title.\n"
                "- 5–12 words.\n"
                "- If missing, infer the most appropriate title.\n\n"

                "description:\n"
                "- Refined version of the input.\n"
                "- 2–4 concise sentences.\n"
                "- Preserve original intent and remove noise.\n\n"

                "audience:\n"
                "- Must be specific and domain-appropriate.\n"
                "- Avoid vague terms like 'everyone' or 'general audience'.\n\n"

                 "mode:\n"
                "- closed_book dont need research.\n"
                "- open_book need research and use only external sources.\n"
                "- hybrid need research but can use both external sources and model's existing knowledge.\n"

                "tone:\n"
                "- 1–3 words maximum.\n"
                "- Must match the intent of the description.\n\n"

                "research fields:\n"
                "- Determine whether additional research is needed to complete the blog.\n"
                "- Set require_research to true only when recent facts, statistics, or "
                "external information are necessary.\n"
                "- Choose research_mode from {closed_book, hybrid, open_book} based on "
                "the level of external information required.\n"
                "- Provide 3–6 concise research_queries only when research_mode is "
                "hybrid or open_book.\n"
                "- Otherwise, return an empty list for research_queries.\n\n"

                "IMPORTANT:\n"
                "- Infer all values only from the provided input.\n"
                "- If uncertain, choose the safest neutral domain-appropriate option.\n\n"

                "OUTPUT RULES:\n"
                "- Return structured data matching ResearchSchema exactly.\n"
                "- Do not include markdown, comments, or extra text."
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
    evidence: List[EvidenceSchema]
) -> list:
    
    evidence_dict= [e.model_dump() for e in evidence]

    return [
        SystemMessage(
            content=(
                "You are a professional blog strategist and audience advocate.\n"
                "Your task is to create a clear, structured plan for a blog post.\n\n"

                "CORE PRINCIPLE:\n"
                "- Adapt depth, vocabulary, and examples based on the provided audience and tone.\n"
                "- Do NOT assume the topic is technical unless explicitly indicated.\n"
                "- The plan must feel natural for the topic domain (technical, business, lifestyle, etc.).\n\n"

                "EVIDENCE USAGE RULES:\n"
                "- You are provided with curated evidence derived from web research.\n"
                "- Use evidence to guide section emphasis and practical relevance.\n"
                "- Do NOT quote evidence verbatim.\n"
                "- Do NOT invent facts beyond the evidence.\n"
                "- If evidence is weak or speculative, deprioritize it.\n\n"

                "HARD REQUIREMENTS:\n"
                "- Create 5–7 sections total.\n"
                "- Exactly ONE section must be 'common_mistakes'.\n"
                "- Section IDs must start at 1 and be sequential.\n"
                "- Each section must follow TaskSchema exactly.\n\n"
                "- Section types MUST be one of:\n"
                "- intro, core, examples, checklist, common_mistakes, conclusion\n"
                "- Do NOT invent new section_type values.\n"
                "- Use \"examples\" for application-oriented or practical sections.\n"

                "QUALITY RULES:\n"
                "- Sections must logically progress from introduction → core ideas → application → conclusion.\n"
                "- Bullets must be concrete and useful for the target audience.\n"
                "- Adjust complexity to audience expertise level.\n"
                "- Include examples, tips, or practical guidance when appropriate.\n\n"

                "Output rules:\n"
                "- Output ONLY valid JSON matching PlanSchema.\n"
                "- No markdown, comments, or explanations."
            )
        ),
        HumanMessage(
            content=(
                f"Blog topic:\n{topic}\n\n"
                f"Blog description:\n{description}\n\n"
                f"Target audience:\n{audience}\n\n"
                f"Writing tone:\n{tone}\n\n"
                f"Evidence from research (if any):\n"
                f"Available evidence from research:\n{evidence_dict}"
            )
        ),
    ]


# worker prompt to generate each section of the blog as per task in plan
def worker_prompt(
    task: TaskSchema,
    blog_topic: str,
    plan: PlanSchema,
    audience: str,
    tone: str,
    evidence: List[EvidenceSchema],
) -> list:

    bullets_text = "\n".join(f"- {b}" for b in task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title}: {e.url}"
            for e in evidence[:6]
        )

    return [
        SystemMessage(
            content=(
                "You are a professional blog writer skilled at adapting to any domain and audience.\n"
                "Write EXACTLY ONE section of the blog.\n\n"

                "OUTPUT FORMAT (MANDATORY):\n"
                "- The response MUST be valid Markdown.\n"
                "- The response MUST be suitable for saving directly as a `.md` file.\n"
                "- Output ONLY the section content.\n"
                "- Do NOT include explanations, comments, or chat-style text.\n\n"

                "EVIDENCE USAGE RULES:\n"
                "- You are provided with external evidence derived from web research.\n"
                "- You MAY use evidence to ground context, emphasis, or examples when relevant.\n"
                "- When evidence is used, reference it INLINE using natural language.\n"
                "- Example formats:\n"
                "  - (recent South Pole Telescope observations)\n"
                "  - (MIT cosmology research)\n"
                "  - (neutrino studies in 2024)\n"
                "- ADD this url from evidence like this [South Pole Telescope](https://news.uchicago.edu/story/latest-data-south-pole-telescope...) if evidence is correct and provided"
                "- Do NOT invent facts, numbers, institutions, dates, or findings.\n"
                "- If evidence is not relevant to the section goal, ignore it completely.\n\n"

                "CONTENT RULES:\n"
                "- Follow the Goal exactly.\n"
                "- Cover ALL bullets in the given order.\n"
                "- Stay within ±15% of the target word count.\n"
                "- Prefer explanations, metaphors, and implications over raw facts.\n"
                "- Mention trade-offs or uncertainties only when they naturally fit the topic.\n\n"

                "STYLE RULES:\n"
                "- Start with: ## <Section Title>\n"
                "- Use short paragraphs and lists where helpful.\n"
                "- Avoid fluff, repetition, meta commentary, or explicit citations."
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
                f"Target words: {task.target_words}\n\n"

                f"Bullets to cover:\n{bullets_text}\n\n"
                f"Available evidence (use selectively):\n{evidence_text}"
            )
        ),
    ]



# prmopt to give structured evidence List from raw result_list_dict
def get_evidence_research_prompt(result_list_dict: List[dict]) -> list:
    return [
        SystemMessage(
            content=(
                "You are a research assistant tasked with extracting relevant evidence from raw search results.\n\n"

                "INPUT:\n"
                "- A list of search result dictionaries, each containing fields like title, content, source, url, and published_date.\n\n"

                "TASK:\n"
                "- Analyze the raw search results and extract a concise list of evidence items that are directly relevant to the research queries.\n"
                "- Each evidence item should include the title, a brief summary of the content (if applicable), the source name, the URL, and the published date (if available).\n"
                "- Focus on relevance and credibility when selecting evidence.\n\n"

                "OUTPUT:\n"
                "- Return a structured list of evidence items in JSON format matching the EvidenceSchema.\n"
                "- Do NOT include any information that is not directly supported by the input data."
            )
        ),
        HumanMessage(
            content=(
                f"Raw search results:\n{result_list_dict}"
            )
        ),
    ]