from models import structured_output_model, get_generation_model,research_structured_output_model, structured_output_model_research
from states import BlogState, EvidencePackSchema, PlanSchema, ResearchSchema
from prompts import get_blog_planning_prompt, worker_prompt,get_router_prompt, get_evidence_research_prompt
from typing import Dict, List, cast
from langgraph.types import Send
from langgraph.graph import END


from utils import normalize_tavily_results, perform_research, safe_filename

#  1. get the blog_topic and other details from initial state

async def router_node(state:BlogState) -> dict:
    try:
        print("Router_node : Extracting details from blog description...\n")
        # 1. Get the prompt for detail extraction
        prompt = get_router_prompt(state.blog_description)
        # 2. call the structured output model to extract details from blog description
        response = cast(ResearchSchema, await research_structured_output_model.ainvoke(prompt))

        print("Extracted details from blog description :\n")
        print(f"Topic: {response.topic}") #type: ignore
        print(f"Description: {response.description}") #type: ignore
        print(f"Audience: {response.audience}") #type: ignore
        print(f"Tone: {response.tone}") #type: ignore
        print(f"Require Research: {response.require_research}") #type: ignore
        print(f"Research Mode: {response.research_mode}") #type: ignore
        print(f"Research Queries: {[query for query in response.research_queries]}\n") #type: ignore


        #  return the extracted details to update the state
        print("Router_node : Detail extraction complete.\n")
        return {
            'blog_topic': response.topic, #type: ignore
            'blog_description': response.description, #type: ignore
            'audience': response.audience,  #type: ignore
            'tone': response.tone,  #type: ignore
            'require_research': response.require_research,  #type: ignore
            'research_mode': response.research_mode,  #type: ignore
            'research_queries': response.research_queries  #type: ignore
            }
    except Exception as e:

        print(f"Research_node : Error in research_node: {e}")
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

    result_list_dict: List[Dict] = []

    for query in queries:
        result = await perform_research(query)
        # result_list_dict.append(result) #type: ignore
        result_list_dict.extend(result.get("results", []))

    if not result_list_dict:
        return {'evidence': []}
    

    # normalize the results into a consistent format for the reducer to consume
    normalized_results = normalize_tavily_results(result_list_dict)

    # get the prompt for evidence extraction from normalized research results
    evidence_research_prompt = get_evidence_research_prompt(normalized_results) 

    # call the structured output model to extract evidence from research results
    response = cast(EvidencePackSchema, await structured_output_model_research.ainvoke(evidence_research_prompt))

    print("Research_node : Research complete. Evidence collected:\n")
    # print(response.evidence) #type: ignore

    return {'evidence': response.evidence} #type: ignore





#  3. Orchestration logic for the blog planning process
async def orchestrator(state:BlogState) -> Dict:
    try:
        print("Orchestrator : Generating blog plan...\n")
        # 1. Get the prompt for blog planning
        blog_description = state.blog_description
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
    

