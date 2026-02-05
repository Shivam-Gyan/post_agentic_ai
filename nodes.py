from models import structured_output_model, generation_model,detail_structured_output_model
from states import BlogState,PlanSchema,TaskSchema,DetailsSchema
from prompts import get_blog_planning_prompt, worker_prompt,get_detail_extraction_prompt
from typing import Dict, Any, cast
from langgraph.types import Send
from pathlib import Path
import asyncio

from utils import safe_filename

#  1. get the blog_topic and other details from initial state

async def detail_node(state:BlogState) -> dict:
    try:
        # 1. Get the prompt for detail extraction
        prompt = get_detail_extraction_prompt(state.blog_description)
        # 2. call the structured output model to extract details from blog description
        response = cast(DetailsSchema, await detail_structured_output_model.ainvoke(prompt))

        print("Extracted details from blog description:", response)
        #  return the extracted details to update the state
        return {
            'blog_topic': response.topic, #type: ignore
            'blog_description': response.description, #type: ignore
            'audience': response.audience,  #type: ignore
            'tone': response.tone   
            }
    except Exception as e:
        print(f"Error in detail_node: {e}")
        raise e 

#  2. Orchestration logic for the blog planning process
async def generate_blog_plan(state:BlogState) -> Dict:
    try:
        # 1. Get the prompt for blog planning
        blog_decription = state.blog_description
        blog_topic = state.blog_topic
        blog_audience = state.audience
        blog_tone = state.tone

        prompt = get_blog_planning_prompt(blog_topic, blog_decription, blog_audience, blog_tone)

        # 2. call the structured output model to generate the plan
        response = await structured_output_model.ainvoke(prompt)

        print("Raw response from model:\n\n", response)

        return {'plan': response }
    except Exception as e:
        print(f"Error in generate_blog_plan: {e}")
        raise e


#  3. intermediate function between orchestrator and workers
# Now we define the fanout function for the node
# which will create multiple worker for as per task in plan 
# after the plan generated 
def fanout(state:BlogState):
    try:
        workers = []
        for task in state.plan.tasks:
            workers.append(
                Send("worker",{
                    'task':task,
                    'blog_topic':state.blog_topic,
                    'plan':state.plan,
                    'audience':state.audience,
                    'tone':state.tone
                })
            )

        return workers
    except Exception as e:
        print(f"Error in fanout: {e}")
        raise e

#  4. actual generation of each task seggregated by worker will executed by worker node
async def worker(payload:dict) -> dict:
    try:
        # extracting whole payload
        task = payload['task']
        blog_topic = payload['blog_topic']
        plan = payload['plan']
        audience = payload['audience']
        tone = payload['tone']

        # get the prompt for worker 
        prompt = worker_prompt(task, blog_topic, plan, audience, tone)

        # generate the content for the task using generation model
        response_msg = await generation_model.ainvoke(prompt)
        response = response_msg.content
        # print(f"Worker response for task {task.id} - {task.title} : \n\n", response)
        return {'sections':[response]}
    except Exception as e:
        print(f"Error in worker node: {e}")
        raise e

#  5. reducer to aggregate all sections from workers into final blog
async def reducer(state:BlogState):
    try:
        title = state.plan.blog_title
        blog = "\n\n".join(state.sections)

        final_blog = f"# {title}\n\n{blog}"

        file_name = safe_filename(title)
        print(f"Saving final blog to {file_name}...\n")

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(final_blog)


        return {'final_blog': final_blog}
    except Exception as e:
        print(f"Error in reducer: {e}")
        raise e