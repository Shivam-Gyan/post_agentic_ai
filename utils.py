from typing import List, Dict, TypeVar, Callable, Awaitable
from langchain_tavily.tavily_search import TavilySearch
from markdown import markdown
from bs4 import BeautifulSoup
import os
import re
import asyncio
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Utility function to create a safe filename from a blog title
def safe_filename(title: str) -> str:
    # lower case
    name = title.lower()

    # replace spaces with underscore
    name = name.replace(" ", "_")

    # remove anything that's NOT a-z, 0-9, _ or -
    name = re.sub(r"[^a-z0-9_-]", "", name)

    # avoid empty names
    if not name:
        name = "blog"

    return f"{name}.md"

#  perform tavily search for a given query and return the results
async def perform_research(query: str, timeout: int = 15):
    search_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=2, search_depth="basic")  # type: ignore
    try:
        response = await asyncio.wait_for(search_tool.ainvoke({"query": query}), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        print(f"perform_research: Tavily timed out for query '{query}' after {timeout}s — returning empty results")
        return {"results": []}

# normalizing the research results into a consistent format for the reducer to consume
def normalize_tavily_results(results: List[Dict]) -> List[Dict]:
    normalized = []

    for r in results:
        content = r.get("content") or ""
        normalized.append({
            "title": r.get("title"),
            "content": content[:300],  # Truncate to prevent Groq function-calling failures
            "url": r.get("url"),
        })  

    return normalized

# utils/retry.py



T = TypeVar("T")

async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    backoff: float = 1.5,
    fallback: T,
    label: str = "task",
) -> T:
    """
    Retry an async callable up to max_retries times with linear back-off.
    
    - ValueError (validation errors) → fail immediately, return fallback
    - Any other exception            → retry with back-off
    - All retries exhausted          → return fallback
    
    Args:
        fn:          Zero-argument async callable to retry  →  lambda: model.ainvoke(prompt)
        max_retries: Max number of attempts (default 3)
        backoff:     Seconds multiplier per attempt (1.5 → 1.5s, 3s, 4.5s)
        fallback:    Value to return when all retries fail
        label:       Human-readable name shown in logs
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = await fn()
            if attempt > 1:
                logger.info("with_retry: '%s' succeeded on attempt %d", label, attempt)
            return result

        except ValueError as ve:
            logger.error("with_retry: '%s' validation error (not retrying). error=%s", label, ve)
            return fallback

        except Exception as e:
            last_error = e
            logger.warning(
                "with_retry: '%s' failed (attempt %d/%d). error=%s",
                label, attempt, max_retries, e,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff * attempt)

    logger.error(
        "with_retry: '%s' all %d attempts exhausted. last_error=%s",
        label, max_retries, last_error,
    )
    return fallback

# parser mode 

def parse_mode(user_input: str):
    if ":" not in user_input:
        return None, user_input

    prefix, content = user_input.split(":", 1)
    prefix = prefix.strip().lower()
    content = content.strip()

    allowed_modes = {"chat", "generate", "refine", "publish"}

    if prefix in allowed_modes:
        return prefix, content

    return None, user_input





def strip_markdown(md_text: str) -> str:
    """Convert markdown to plain readable text for TTS."""

    # 1. Convert markdown → HTML
    html = markdown(md_text)

    # 2. Parse HTML → plain text
    soup = BeautifulSoup(html, "html.parser")
    plain = soup.get_text(separator=" ")

    # 3. Clean up leftover symbols and whitespace
    plain = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', plain)  # links → just label
    plain = re.sub(r'`+', '', plain)                          # backticks
    plain = re.sub(r'#+\s*', '', plain)                       # headings hashes
    plain = re.sub(r'\*+', '', plain)                         # bold/italic stars
    plain = re.sub(r'_{1,2}', '', plain)                      # underscores
    plain = re.sub(r'~{1,2}', '', plain)                      # strikethrough
    plain = re.sub(r'>\s*', '', plain)                        # blockquotes
    plain = re.sub(r'-{3,}|={3,}', '', plain)                 # horizontal rules
    plain = re.sub(r'\n{3,}', '\n\n', plain)                  # excess newlines
    plain = re.sub(r' {2,}', ' ', plain)                      # excess spaces

    return plain.strip()

MAX_TTS_CHARS = 1000  # stay safely under Groq's 4000 limit

def truncate_to_limit(text: str, limit: int = MAX_TTS_CHARS) -> str:
    """Truncate at sentence boundary to stay under TTS char limit."""
    if len(text) <= limit:
        return text
    
    truncated = text[:limit]
    # try to end at a sentence boundary
    last_period = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_period > limit // 2:  # only use boundary if it's not too early
        return truncated[:last_period + 1]
    
    return truncated.strip()


# if __name__ == "__main__":
#     research_results = asyncio.run(perform_research("Oracle trending news"))
#     print(research_results)



# blog = '''
# # The Impact of Generative AI and Large Language Models on Machine Learning Evolution

# ## Introduction to Generative AI and ML Convergence

# The landscape of machine learning (ML) is undergoing a significant transformation with the advent of generative AI and large language models (LLMs). As we navigate through 2025, these technologies are not only redefining the capabilities of ML but also accelerating its evolution. Generative AI, a subset of AI that focuses on generating new, synthetic data that resembles existing data, has seen substantial advancements. This includes improved annotation techniques, the use of synthetic data, human-in-the-loop validation, and enhanced data management practices [Generative AI Trends: 2025 Market Report - Clickworker].

# Large language models (LLMs), a type of generative AI, have been at the forefront of this evolution. These models are capable of understanding and generating human-like language, enabling a wide range of applications from natural language processing to content creation. However, their development and application also raise significant ethical challenges, including AI hallucinations, information bias, privacy risks, and transparency deficiencies [Ethical Considerations and Fundamental Principles of Large ...].

# A key distinction between traditional ML and generative AI approaches lies in their objectives and methodologies. Traditional ML focuses on predictive modeling, where the goal is to forecast outcomes based on historical data. In contrast, generative AI aims to create new data or content, pushing the boundaries of what ML can achieve. Despite the potential of generative AI, it currently lags behind traditional ML models in forecasting accuracy, highlighting the complementary nature of these approaches [Generative AI vs Traditional ML Models in Forecasting].

# As we explore the convergence of ML and generative AI, a central question emerges: How are these technologies accelerating the evolution of ML, and what implications does this have for businesses and technology innovation? By examining the intersections and synergies between ML and generative AI, we can better understand the future trajectory of this rapidly evolving field. The MIT Sloan article provides insights into the applications of machine learning and generative AI in 2025, emphasizing their roles in business and technology innovation [Machine learning and generative AI: What are they good for in 2025?].

# ## Foundational Shifts in Model Training

# The integration of Large Language Models (LLMs) and generative AI into machine learning (ML) workflows is driving significant foundational shifts in model training. A key area of advancement is the generation and utilization of synthetic data. According to the [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), improved annotation techniques and synthetic data usage are expected to play crucial roles in 2025. Synthetic data generation enables the creation of large datasets that can be used to train LLMs, potentially reducing the need for manually annotated data and alleviating data scarcity issues.

# Another critical development is the incorporation of human-in-the-loop validation versus automated model training pipelines. Human-in-the-loop validation allows for more accurate and context-sensitive model outputs by enabling human oversight and correction during the training process. In contrast, automated pipelines prioritize efficiency and scalability but may compromise on accuracy and reliability. The choice between these approaches depends on the specific application and the trade-offs between accuracy, cost, and deployment speed.

# Compliance-driven data management frameworks are also becoming increasingly important for LLMs. As highlighted in the [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), enhanced data management practices are a key trend in 2025, driven in part by the need to comply with evolving AI regulations. Effective data management ensures that LLMs are trained on high-quality, diverse data while minimizing risks related to data privacy and bias. This shift towards robust data governance is essential for the responsible development and deployment of LLMs in real-world applications. By prioritizing data quality, compliance, and human-centric validation, organizations can harness the full potential of generative AI and LLMs while mitigating associated risks.

# ## Ethical Challenges in LLM Development

# The development and deployment of Large Language Models (LLMs) have introduced several ethical challenges that need to be addressed to ensure responsible AI innovation. One of the primary concerns is the phenomenon of AI hallucinations, where models generate false or misleading information. This issue is compounded by information bias, which can lead to skewed or inaccurate outputs. A study published in [JMIR's ethical principles study](https://www.jmir.org/2024/1/e60083/) highlights these challenges and emphasizes the need for robust governance frameworks to mitigate them.

# Another critical concern is the risk of privacy breaches in synthetic data generation. As LLMs increasingly rely on synthetic data for training and validation, ensuring the privacy and security of this data becomes paramount. The [2025 Market Report by Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/) notes the importance of improved annotation techniques and human-in-the-loop validation in addressing these issues.

# To address these challenges, it is essential to establish transparent model outputs and robust governance frameworks. This can be achieved through several measures:

# * **Model interpretability**: Developing techniques to explain and interpret model outputs can help identify potential biases and hallucinations.
# * **Data quality and validation**: Ensuring the accuracy and reliability of training data, as well as implementing robust validation procedures, can help mitigate the risks associated with synthetic data generation.
# * **Regulatory compliance**: Adhering to emerging AI regulations and guidelines, such as those outlined in the [2025 Market Report by Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), can help ensure that LLMs are developed and deployed responsibly.

# By prioritizing transparency, accountability, and regulatory compliance, we can harness the potential of LLMs while minimizing their risks and ensuring that they are developed and deployed in a responsible and ethical manner.

# ## Common Mistakes in Generative AI Implementation

# As organizations increasingly adopt generative AI and large language models (LLMs), it's essential to acknowledge common pitfalls that can hinder successful implementation. One prevalent misconception is that generative AI can serve as a universal forecasting solution. However, [Generative AI vs Traditional ML Models in Forecasting](https://www.facebook.com/groups/698593531630485/posts/1056751559148012/) notes that generative AI currently lags behind traditional machine learning models in forecasting accuracy. This disparity underscores limitations in its predictive capabilities compared to established methods.

# Another mistake is the over-reliance on synthetic data without domain validation. While [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/) highlights improved annotation techniques and synthetic data usage as key 2025 advancements, it's crucial to validate synthetic data within specific domains to ensure its reliability and accuracy.

# Moreover, organizations must be aware of the forecasting accuracy gaps between generative AI and traditional ML models. [Machine learning and generative AI: What are they good for in 2025?](https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-and-generative-ai-what-are-they-good-for) explores applications of machine learning and generative AI in 2025, but it's essential to recognize that generative AI is not a replacement for traditional ML models in all scenarios.

# Finally, as generative AI and LLMs continue to evolve, it's vital to address ethical challenges, such as AI hallucinations, information bias, privacy risks, and transparency deficiencies [Ethical Considerations and Fundamental Principles of Large ...](https://www.jmir.org/2024/1/e60083/). By acknowledging these common mistakes and challenges, organizations can ensure a more effective and responsible implementation of generative AI and LLMs.

# ## Future Trajectory of ML-LLM Integration

# As we look ahead to the future of machine learning (ML) and large language models (LLMs) integration, several emerging trends are poised to shape the landscape. One key area of development is in **multi-modal LLM training pipelines**. These pipelines, which involve training models on diverse data types such as text, images, and audio, are expected to become increasingly prevalent. This shift towards multi-modality will enable LLMs to better understand and interact with the world, leading to more sophisticated applications in areas like business and technology innovation [Machine learning and generative AI: What are they good for in 2025?](https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-and-generative-ai-what-are-they-good-for).

# Another significant trend is the growing use of **synthetic data** in LLM training. Synthetic data, which is artificially generated data, offers a promising solution to data scarcity and quality issues. However, its usage also raises regulatory concerns. As noted in the [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), we can expect to see increased **regulatory impacts on synthetic data usage** in the coming years. This will likely involve stricter guidelines on data generation, usage, and sharing to ensure transparency and accountability.

# The rapid development of AI technologies also poses significant **technical debt challenges**. As LLMs continue to evolve, the need for robust governance frameworks to address issues like AI hallucinations, information bias, and privacy risks becomes increasingly pressing [Ethical Considerations and Fundamental Principles of Large ...](https://www.jmir.org/2024/1/e60083/). Furthermore, the limitations of generative AI in predictive capabilities, as highlighted in a Facebook post analysis comparing generative AI to traditional ML models in forecasting [Generative AI vs Traditional ML Models in Forecasting](https://www.facebook.com/groups/698593531630485/posts/1056751559148012/), underscore the importance of careful planning and management in AI development.

# In conclusion, the future trajectory of ML-LLM integration will be shaped by emerging trends in multi-modal LLM training pipelines, regulatory impacts on synthetic data usage, and technical debt challenges. As senior machine learning engineers and data scientists, it is essential to stay informed about these developments and to prioritize responsible AI development practices to ensure the benefits of these technologies are realized while minimizing their risks.

# '''
   