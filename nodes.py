from models import structured_output_model, get_generation_model,research_structured_output_model, structured_output_model_research,intent_structured_output_model,refine_feedback_output_model
from states import BlogState, EvidencePackSchema, PlanSchema, ResearchSchema
from prompts import get_blog_planning_prompt, get_feedback_prompt, worker_prompt,get_router_prompt, get_evidence_research_prompt, get_intent_detection_prompt
from typing import Dict, List, cast
from langgraph.types import Send, interrupt
from langgraph.errors import GraphInterrupt
from langgraph.graph import END
from utils import parse_mode
import httpx


from utils import normalize_tavily_results, perform_research, safe_filename


# Intent detection node to detetrmine ['generate', 'refine', 'chat']

async def intent_detection_node(state: BlogState) :

    try:

        # response = await intent_structured_output_model.ainvoke(get_intent_detection_prompt(state.user_query)) #type: ignore
        # print(f"Intent detection response: {response}")

        mode, cleaned_query = parse_mode(state.user_query)

        if not mode and not cleaned_query:
            raise ValueError("Intent detection failed to parse mode and query")

        state.user_query = cleaned_query
        # state = app.invoke(state)
        
        blog = '''
# The Impact of Generative AI and Large Language Models on Machine Learning Evolution

## Introduction to Generative AI and ML Convergence

The landscape of machine learning (ML) is undergoing a significant transformation with the advent of generative AI and large language models (LLMs). As we navigate through 2025, these technologies are not only redefining the capabilities of ML but also accelerating its evolution. Generative AI, a subset of AI that focuses on generating new, synthetic data that resembles existing data, has seen substantial advancements. This includes improved annotation techniques, the use of synthetic data, human-in-the-loop validation, and enhanced data management practices [Generative AI Trends: 2025 Market Report - Clickworker].

Large language models (LLMs), a type of generative AI, have been at the forefront of this evolution. These models are capable of understanding and generating human-like language, enabling a wide range of applications from natural language processing to content creation. However, their development and application also raise significant ethical challenges, including AI hallucinations, information bias, privacy risks, and transparency deficiencies [Ethical Considerations and Fundamental Principles of Large ...].

A key distinction between traditional ML and generative AI approaches lies in their objectives and methodologies. Traditional ML focuses on predictive modeling, where the goal is to forecast outcomes based on historical data. In contrast, generative AI aims to create new data or content, pushing the boundaries of what ML can achieve. Despite the potential of generative AI, it currently lags behind traditional ML models in forecasting accuracy, highlighting the complementary nature of these approaches [Generative AI vs Traditional ML Models in Forecasting].

As we explore the convergence of ML and generative AI, a central question emerges: How are these technologies accelerating the evolution of ML, and what implications does this have for businesses and technology innovation? By examining the intersections and synergies between ML and generative AI, we can better understand the future trajectory of this rapidly evolving field. The MIT Sloan article provides insights into the applications of machine learning and generative AI in 2025, emphasizing their roles in business and technology innovation [Machine learning and generative AI: What are they good for in 2025?].

## Foundational Shifts in Model Training

The integration of Large Language Models (LLMs) and generative AI into machine learning (ML) workflows is driving significant foundational shifts in model training. A key area of advancement is the generation and utilization of synthetic data. According to the [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), improved annotation techniques and synthetic data usage are expected to play crucial roles in 2025. Synthetic data generation enables the creation of large datasets that can be used to train LLMs, potentially reducing the need for manually annotated data and alleviating data scarcity issues.

Another critical development is the incorporation of human-in-the-loop validation versus automated model training pipelines. Human-in-the-loop validation allows for more accurate and context-sensitive model outputs by enabling human oversight and correction during the training process. In contrast, automated pipelines prioritize efficiency and scalability but may compromise on accuracy and reliability. The choice between these approaches depends on the specific application and the trade-offs between accuracy, cost, and deployment speed.

Compliance-driven data management frameworks are also becoming increasingly important for LLMs. As highlighted in the [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), enhanced data management practices are a key trend in 2025, driven in part by the need to comply with evolving AI regulations. Effective data management ensures that LLMs are trained on high-quality, diverse data while minimizing risks related to data privacy and bias. This shift towards robust data governance is essential for the responsible development and deployment of LLMs in real-world applications. By prioritizing data quality, compliance, and human-centric validation, organizations can harness the full potential of generative AI and LLMs while mitigating associated risks.

## Ethical Challenges in LLM Development

The development and deployment of Large Language Models (LLMs) have introduced several ethical challenges that need to be addressed to ensure responsible AI innovation. One of the primary concerns is the phenomenon of AI hallucinations, where models generate false or misleading information. This issue is compounded by information bias, which can lead to skewed or inaccurate outputs. A study published in [JMIR's ethical principles study](https://www.jmir.org/2024/1/e60083/) highlights these challenges and emphasizes the need for robust governance frameworks to mitigate them.

Another critical concern is the risk of privacy breaches in synthetic data generation. As LLMs increasingly rely on synthetic data for training and validation, ensuring the privacy and security of this data becomes paramount. The [2025 Market Report by Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/) notes the importance of improved annotation techniques and human-in-the-loop validation in addressing these issues.

To address these challenges, it is essential to establish transparent model outputs and robust governance frameworks. This can be achieved through several measures:

* **Model interpretability**: Developing techniques to explain and interpret model outputs can help identify potential biases and hallucinations.
* **Data quality and validation**: Ensuring the accuracy and reliability of training data, as well as implementing robust validation procedures, can help mitigate the risks associated with synthetic data generation.
* **Regulatory compliance**: Adhering to emerging AI regulations and guidelines, such as those outlined in the [2025 Market Report by Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), can help ensure that LLMs are developed and deployed responsibly.

By prioritizing transparency, accountability, and regulatory compliance, we can harness the potential of LLMs while minimizing their risks and ensuring that they are developed and deployed in a responsible and ethical manner.

## Common Mistakes in Generative AI Implementation

As organizations increasingly adopt generative AI and large language models (LLMs), it's essential to acknowledge common pitfalls that can hinder successful implementation. One prevalent misconception is that generative AI can serve as a universal forecasting solution. However, [Generative AI vs Traditional ML Models in Forecasting](https://www.facebook.com/groups/698593531630485/posts/1056751559148012/) notes that generative AI currently lags behind traditional machine learning models in forecasting accuracy. This disparity underscores limitations in its predictive capabilities compared to established methods.

Another mistake is the over-reliance on synthetic data without domain validation. While [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/) highlights improved annotation techniques and synthetic data usage as key 2025 advancements, it's crucial to validate synthetic data within specific domains to ensure its reliability and accuracy.

Moreover, organizations must be aware of the forecasting accuracy gaps between generative AI and traditional ML models. [Machine learning and generative AI: What are they good for in 2025?](https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-and-generative-ai-what-are-they-good-for) explores applications of machine learning and generative AI in 2025, but it's essential to recognize that generative AI is not a replacement for traditional ML models in all scenarios.

Finally, as generative AI and LLMs continue to evolve, it's vital to address ethical challenges, such as AI hallucinations, information bias, privacy risks, and transparency deficiencies [Ethical Considerations and Fundamental Principles of Large ...](https://www.jmir.org/2024/1/e60083/). By acknowledging these common mistakes and challenges, organizations can ensure a more effective and responsible implementation of generative AI and LLMs.

## Future Trajectory of ML-LLM Integration

As we look ahead to the future of machine learning (ML) and large language models (LLMs) integration, several emerging trends are poised to shape the landscape. One key area of development is in **multi-modal LLM training pipelines**. These pipelines, which involve training models on diverse data types such as text, images, and audio, are expected to become increasingly prevalent. This shift towards multi-modality will enable LLMs to better understand and interact with the world, leading to more sophisticated applications in areas like business and technology innovation [Machine learning and generative AI: What are they good for in 2025?](https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-and-generative-ai-what-are-they-good-for).

Another significant trend is the growing use of **synthetic data** in LLM training. Synthetic data, which is artificially generated data, offers a promising solution to data scarcity and quality issues. However, its usage also raises regulatory concerns. As noted in the [Generative AI Trends: 2025 Market Report - Clickworker](https://www.clickworker.com/customer-blog/generative-ai-trends/), we can expect to see increased **regulatory impacts on synthetic data usage** in the coming years. This will likely involve stricter guidelines on data generation, usage, and sharing to ensure transparency and accountability.

The rapid development of AI technologies also poses significant **technical debt challenges**. As LLMs continue to evolve, the need for robust governance frameworks to address issues like AI hallucinations, information bias, and privacy risks becomes increasingly pressing [Ethical Considerations and Fundamental Principles of Large ...](https://www.jmir.org/2024/1/e60083/). Furthermore, the limitations of generative AI in predictive capabilities, as highlighted in a Facebook post analysis comparing generative AI to traditional ML models in forecasting [Generative AI vs Traditional ML Models in Forecasting](https://www.facebook.com/groups/698593531630485/posts/1056751559148012/), underscore the importance of careful planning and management in AI development.

In conclusion, the future trajectory of ML-LLM integration will be shaped by emerging trends in multi-modal LLM training pipelines, regulatory impacts on synthetic data usage, and technical debt challenges. As senior machine learning engineers and data scientists, it is essential to stay informed about these developments and to prioritize responsible AI development practices to ensure the benefits of these technologies are realized while minimizing their risks.

'''
        
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
        return 'refine_structured_output_node'
    elif mode == "chat":
        return 'chat_node'
    elif mode == "publish":
        return 'publish_node'
    else:
        # default to router_node for generation if mode is unrecognized
        return 'router_node'


async def refine_structured_output_model(state: BlogState) -> dict:

    try:
        prev_feedback = state.refinement.feedback[-1] if state.refinement.feedback else "No previous feedback"
        refine_feedabck_prompt = get_feedback_prompt(state.user_query, state.final_blog, prev_feedback)
        response = await refine_feedback_output_model.ainvoke(refine_feedabck_prompt) 
        print(f"Refinement feedback response: {response}")
        return {
            "refinement": {
                "feedback": [response.feedback] # type: ignore
            }
        }
    except Exception as e:
        print(f"Error in refine_structured_output_model: {e}")
        raise e


#  1. get the blog_topic and other details from initial state

async def router_node(state:BlogState) -> dict:
    try:
        print("Router_node : Extracting details from blog description...\n")
        # 1. Get the prompt for detail extraction
        prompt = get_router_prompt(state.user_query)
        # 2. call the structured output model to extract details from blog description
        response = cast(ResearchSchema, await research_structured_output_model.ainvoke(prompt))

        #  return the extracted details to update the state
        print("Router_node : Detail extraction complete.\n")

        return {
            'blog_topic': response.topic, #type: ignore
            'blog_description': response.description, #type: ignore
            'audience': response.audience,  #type: ignore
            'tone': response.tone,  #type: ignore
            'require_research': response.require_research,  #type: ignore
            'research_mode': response.research_mode,  #type: ignore
            'research_queries': response.research_queries, #type: ignore
            'blog_kind': response.blog_kind #type: ignore
            }
    
    except Exception as e:

        print(f"Router_node : Error in router_node: {e}")
        raise e 


# Routing Conditon
def router_condition_func(state: BlogState) -> str:
    if state.require_research:
        return 'research_node'
    else:
        # return END
        return 'orchestrator'


# 2. Research_node
async def research_node(state:BlogState) -> dict:

    # get queries from state
    queries = state.research_queries or []

    raw_result: List[Dict] = []

    for query in queries:
        result = await perform_research(query)
        # result_list_dict.append(result) #type: ignore
        raw_result.extend(result.get("results", []))

    if not raw_result:
        return {'evidence': []}
    
    # 2. Deduplicate by URL (Critical for Production)
    unique_results = {}
    for r in raw_result:
        url = r.get("url")
        if url not in unique_results:
            unique_results[url] = r
    

    deduplicated_list = list(unique_results.values())
    

    # normalize the results into a consistent format for the reducer to consume
    normalized_results = normalize_tavily_results(deduplicated_list)

    # get the prompt for evidence extraction from normalized research results
    evidence_research_prompt = get_evidence_research_prompt(normalized_results) 

    # call the structured output model to extract evidence from research results
    response = cast(EvidencePackSchema, await structured_output_model_research.ainvoke(evidence_research_prompt))

    print("Research_node : Research complete. Evidence collected:\n")

    return {'evidence': response.evidence} #type: ignore



#  3. Orchestration logic for the blog planning process
async def orchestrator(state:BlogState) -> Dict:
    try:
        print("Orchestrator : Generating blog plan...\n")
        # 1. Get the prompt for blog planning
        blog_description = state.user_query
        blog_topic = state.blog_topic
        blog_audience = state.audience
        blog_tone = state.tone
        blog_evidence = state.evidence

        prompt = get_blog_planning_prompt(blog_topic, blog_description, blog_audience, blog_tone, blog_evidence)

        # 2. call the structured output model to generate the plan
        response = cast(PlanSchema, await structured_output_model.ainvoke(prompt))

        # print("Raw response from model:\n\n", response)
        print("Orchestrator : Blog plan generation complete.\n")

        return {'plan': response ,"blog_title": response.blog_title} #type: ignore
    except Exception as e:
        print(f"Orchestrator : Error in generate_blog_plan: {e}")
        raise e



#  4. intermediate function between orchestrator and workers
# Now we define the fanout function for the node
# which will create multiple worker for as per task in plan 
# after the plan generated 

def fanout(state: BlogState) -> List[Send]:
    try:
        if not state.plan or not state.plan.tasks:
            raise ValueError("No plan or tasks found in state")

        tasks = state.plan.tasks
        workers: List[Send] = []

        # batch tasks to reduce GPU calls
        for task in tasks:

            workers.append(
                Send(
                    "worker",
                    {
                        "task": task,
                        "blog_topic": state.blog_topic,
                        "plan": state.plan,
                        "audience": state.audience,
                        "tone": state.tone,
                        'evidence' : state.evidence  
                    },
                )
            )

        return workers

    except Exception as e:
        # this will surface clearly in LangGraph logs
        raise RuntimeError(f"[fanout] failed: {e}") from e

#  5. actual generation of each task seggregated by worker will executed by worker node
async def worker(payload: dict) -> dict:
    try:
        # ---- Validate payload ----
        task = payload["task"]
        if not task:
            raise ValueError("Worker received empty task batch")

        print(f"Worker : Generating sections for task: {getattr(task, 'title', 'unknown')}...\n")

        blog_topic = payload["blog_topic"]
        plan = payload["plan"]
        audience = payload["audience"]
        tone = payload["tone"]
        evidence = payload.get("evidence", [])

        if not blog_topic or not plan:
            raise ValueError("Missing blog topic or plan in worker payload")
        
        if not audience or not tone:
            raise ValueError("Missing audience or tone in worker payload")

        if not evidence:
            raise ValueError("Missing evidence in worker payload")

        # ---- Build prompt for grouped tasks ----
        prompt = worker_prompt(
            task=task,
            blog_topic=blog_topic,
            plan=plan,
            audience=audience,
            tone=tone,
            evidence=evidence
        )

        # ---- Model inference (GPU-bound) ----
        model = get_generation_model()
        response_msg = await model.ainvoke(prompt)

        content = response_msg.content.strip() #type: ignore
        if not content:
            raise ValueError("Empty response from model")
        
        print(f"Worker : Section generation complete for task: {getattr(task, 'title', 'unknown')}.\n")

        return {"sections": [content]}

    except Exception as e:
        # ---- Graceful fallback ----
        
        task = payload["task"]

        print(f"Worker : Error generating sections for task: {getattr(task, 'title', 'unknown')}. Error: {e}\n")

        error_section = (
            f"## ⚠️ Section generation failed\n\n"
            f"**Affected sections:** {getattr(task, 'title', 'unknown')}\n\n"
            f"**Error:** {str(e)}\n\n"
            f"**Trace (truncated):**\n"
        )

        # IMPORTANT:
        # We return a section instead of raising
        # so the reducer and graph can continue
        return {"sections": [error_section]}

#  6. reducer to aggregate all sections from workers into final blog
async def reducer(state:BlogState):
    try:   

        print("Reducer : Aggregating sections from workers...\n")
        title = state.blog_title or "Untitled Blog"
        blog = "\n\n".join(state.sections)

        final_blog = f"# {title}\n\n{blog}"

        file_name = safe_filename(title)
        print(f"Saving final blog to {file_name}...\n")

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(final_blog)

        print(f"Reducer : Final blog aggregation complete and saved to {file_name}.\n")
        return {'final_blog': final_blog}
    except Exception as e:
        print(f"Error in reducer: {e}")
        raise e


#  7. HITL publish node – asks user whether to post blog to Feather Feable
async def publish_node(state: BlogState) -> dict:
    """Human-in-the-loop node: interrupt the graph and wait for the user
    to decide whether to publish the blog to Feather Feable."""
    try:
        print("Publish_node : Waiting for user decision on publishing...\n")

        # ---- Interrupt – control returns to the client ----
        user_response = interrupt({
            "question": "Do you want to publish this blog to Feather Feable blog web app?",
            "options": ["yes", "no"]
        })

        # ---- User chose to publish ----
        if isinstance(user_response, dict) and user_response.get("approved"):
            access_token = user_response.get("access_token", "")
            if not access_token:
                print("Publish_node : No access token provided. Skipping publish.\n")
                return {"publish_result": "❌ Publishing skipped: No access token provided."}

            print("Publish_node : Publishing blog to Feather Feable...\n")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/api/v1/blog/auto-blog",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "title": state.blog_title or "Untitled Blog",
                        "tags": ["ai", "auto-generated"],
                        "des": (state.blog_topic or "")[:250],
                        "markdown": state.final_blog,
                    },
                    timeout=30.0,
                )

            if response.status_code in (200, 201):
                print(f"Publish_node : Blog published successfully! Status: {response.status_code}\n")
                return {"publish_result": f"✅ Blog published successfully! (HTTP {response.status_code})"}
            else:
                print(f"Publish_node : Publishing failed. Status: {response.status_code}\n")
                return {
                    "publish_result": (
                        f"❌ Publishing failed. HTTP {response.status_code} — "
                        f"{response.text[:300]}"
                    )
                }

        # ---- User chose NOT to publish ----
        else:
            print("Publish_node : User skipped publishing.\n")
            return {"publish_result": "⏭️ Publishing skipped by user."}

    except httpx.RequestError as e:
        print(f"Publish_node : Network error during publishing: {e}\n")
        return {"publish_result": f"❌ Network error: {str(e)}"}
    except GraphInterrupt:
        raise  # Let the interrupt propagate to pause the graph
    except Exception as e:
        print(f"Publish_node : Unexpected error: {e}\n")
        return {"publish_result": f"❌ Unexpected error: {str(e)}"}
    

