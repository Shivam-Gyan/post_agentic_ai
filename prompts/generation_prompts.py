from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from states import  EvidenceSchema, PlanSchema, TaskSchema



# prompt to extract details from the initial blog description provided by user
def get_router_prompt(blog_description: str) -> list:
    return [
        SystemMessage(
            content=(
                "### ROLE\n"
                "You are an Expert Content Strategist and Information Architect. Your goal is to transform a raw, "
                "potentially vague user request into a high-fidelity execution blueprint for a top-tier blog post.\n\n"

                "### TASK\n"
                "Analyze the 'Raw Description' and synthesize the metadata required for the ResearchSchema. "
                "Your output determines the quality of the final research and writing phase.\n\n"

                "### STRATEGIC EXTRACTION GUIDELINES\n"
                "1. **Topic & Description**: Elevate the user's intent. If the input is 'AI trends', the topic should be "
                "'The Shift in Generative AI: Key Trends Shaping 2026'. Ensure the description is action-oriented.\n"
                "2. **Audience & Tone**: Identify the 'High-Value Reader'. Instead of 'Developers', use 'Senior DevOps Engineers "
                "looking for efficiency gains'. Match the tone to the audience (e.g., 'Authoritative yet accessible').\n"
                "3. **Research Decision Logic**: \n"
                "   - `require_research`: Set to TRUE if the topic involves real-world data, news, current events (2025-2026), "
                "technical benchmarks, or specific locations (e.g., Greater Noida AQI).\n"
                "   - `research_mode`: \n"
                "       - 'open_book': For time-sensitive data (AQI, Weather, Stocks).\n"
                "       - 'hybrid': For conceptual topics requiring real-world examples.\n"
                "       - 'closed_book': Only for evergreen, creative, or purely philosophical writing.\n"
                "   - `research_queries`: If research is needed, generate 3-6 high-intent search queries. Use advanced operators "
                "like 'latest statistics' or 'technical comparison'.\n"
                "4. **Blog Kind Selection**:\n"
                "   - 'news_roundup': Current events/AQI/Weather.\n"
                "   - 'tutorial': How-to guides.\n"
                "   - 'system_design': Technical architecture.\n\n"

                "### CONSTRAINT\n"
                "Return ONLY the structured JSON data. No conversational filler. No preamble."
            )
        ),
        HumanMessage(
            content=(
                f"Raw blog description provided by the user: '{blog_description}'"
            )
        ),
    ]

def get_blog_planning_prompt(
    topic: str,
    description: str,
    audience: str,
    tone: str,
    evidence: List[EvidenceSchema]
) -> list:
    
    evidence_context = "\n".join([f"- {e.title}: {e.content} (Source: {e.url})" for e in evidence])

    return [
        SystemMessage(
            content=(
                "### ROLE\n"
                "You are a Senior Content Strategist. Your mission is to decompose a blog topic into a high-impact, "
                "logical 'Execution Plan'. Your plan will be used by parallel agents to write individual sections.\n\n"

                "### INPUT CONTEXT\n"
                f"**Main Topic:** {topic}\n"
                f"**Strategic Intent:** {description}\n"
                f"**Target Persona:** {audience}\n"
                f"**Brand Voice:** {tone}\n\n"

                "### PROVIDED RESEARCH (GROUNDING)\n"
                f"{evidence_context}\n\n"

                "### PLANNING ARCHITECTURE RULES\n"
                "1. **Logical Narrative**: The sequence must follow: Hook (Intro) -> Foundational Knowledge (Core) -> "
                "Current Advancements (Core/Examples) -> Practical Pitfalls (Common Mistakes) -> Summary (Conclusion).\n"
                "2. **Evidence Integration**: Assign specific pieces of research to the most relevant tasks. If research "
                "mentions 'Microsoft HSA', that evidence MUST be linked to a task title.\n"
                "3. **Audience Calibration**: \n"
                "   - If Audience is 'Technical': Include tasks for 'require_code' and 'require_citations'.\n"
                "   - If Audience is 'Beginner': Focus on analogies and 'checklist' section types.\n"
                "4. **The 'Common Mistakes' Rule**: You must include exactly one 'common_mistakes' section. Use it to "
                "address myths or frequent errors related to the evidence found.\n\n"

                "### TASK-SPECIFIC INSTRUCTIONS\n"
                "- **Bullets**: Write 3-5 specific sub-instructions for the writer. Avoid generic bullets like 'write intro'. "
                "Use 'Explain the shift from additive to multiplicative attention' instead.\n"
                "- **Word Counts**: Distribute 1500-2500 words across 5-7 tasks. Core sections should have the highest count.\n\n"

                "### OUTPUT FORMAT\n"
                "Return ONLY a JSON object matching PlanSchema. No conversation. No markdown."
            )
        )
    ]


def worker_prompt(
    task: TaskSchema,
    blog_topic: str,
    plan: PlanSchema,
    audience: str,
    tone: str,
    evidence: List[EvidenceSchema],
) -> list:

    bullets_text = "\n".join(f"- {b}" for b in task.bullets)
    
    # Create a simple outline so the worker knows the full story
    outline = "\n".join([f"{t.id}. {t.title}, {t.goal}" for t in plan.tasks])

    # Improved evidence formatting: Include the content/snippet, not just the URL
    # A worker can't cite what it can't read!
    evidence_text = "\n".join(
        f"SOURCE: {e.title}\nURL: {e.url}\nCONTENT: {e.content[:300]}..." #type: ignore
        for e in evidence[:5]
    )

    return [
        SystemMessage(
            content=(
                "### ROLE\n"
                "You are an Elite Technical Writer. Your task is to write ONE specific chapter of a comprehensive blog post. "
                "You must maintain the flow of the overall narrative while strictly adhering to the assigned section's goals.\n\n"

                "### GLOBAL BLOG CONTEXT\n"
                f"**Full Blog Outline:**\n{outline}\n"
                f"**Target Audience:** {audience}\n"
                f"**Writing Tone:** {tone}\n\n"

                "### YOUR ASSIGNED SECTION\n"
                f"**Title:** {task.title}\n"
                f"**Section Type:** {task.section_type}\n"
                f"**Word Count Goal:** {task.target_words} words\n"
                f"**Key Points to Cover:**\n{bullets_text}\n\n"

                "### WRITING CONSTRAINTS\n"
                "1. **No Repetition**: Do not re-introduce the entire topic. Start directly with the substance of your section.\n"
                "2. **Seamless Flow**: Transition naturally. If you are not the 'Intro' section, assume the reader already knows the basics.\n"
                "3. **Markdown Only**: Use H2 (##) for your title. Use bolding for emphasis, but do not use H1 (#).\n"
                "4. **Factual Grounding & Citations**:\n"
                    "- Ground your writing ONLY in the provided research snippets.\n"
                    "- **Citation Format**: When you mention a fact from the evidence, you MUST link it using the 'title' from that specific evidence item as the link text.\n"
                    "- **Example**: If the evidence title is 'Microsoft HSA 2025', write it like: [Microsoft HSA 2025](URL).\n"
                    "- **NEVER** use generic text like '[Source Name]' or '[Link]'. Always use the actual descriptive title of the source provided."
                "### OUTPUT RULES\n"
                "- Output ONLY the Markdown content.\n"
                "- No 'Here is your section' or other conversational filler."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {blog_topic}\n\n"
                f"Research Evidence to include:\n{evidence_text}"
            )
        ),
    ]


def get_evidence_research_prompt(raw_result: List[dict]) -> list:
    return [
        SystemMessage(
            content=(
                "### ROLE\n"
                "You are a Senior Research Analyst. Your task is to synthesize raw search results into a "
                "high-quality, fact-checked Evidence Pack.\n\n"

                "### INPUT DATA\n"
                f"{raw_result}\n\n"

                "### SELECTION CRITERIA\n"
                "1. **Relevance**: Only include items that directly answer the core research questions.\n"
                "2. **Evidence Quality**: Prioritize specific facts, numbers, dates, and authoritative quotes over general marketing fluff.\n"
                "3. **Recency**: For news or technical topics, prioritize the most recent search results.\n"
                "4. **No Redundancy**: If multiple sources provide the same information, select the most detailed one.\n\n"

                "### EXTRACTION RULES\n"
                "- **Grounding**: ONLY use information provided in the input. Do NOT use outside knowledge.\n"
                "- **Content Synthesis**: Summarize the 'content' field into 2-3 dense, informative sentences. Do not just copy-paste.\n"
                "- **Missing Data**: If a URL or Title is missing in the raw data, omit that specific item.\n\n"

                "### OUTPUT INSTRUCTIONS\n"
                "- Return a valid JSON object matching the EvidencePackSchema.\n"
                "- Ensure the 'evidence' list is empty if no relevant information is found.\n"
                "- No preamble, no markdown blocks, just the raw JSON."
            )
        )
    ]
