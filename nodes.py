from models import structured_output_model, generation_model
from states import BlogState,PlanSchema,TaskSchema
from prompts import get_blog_planning_prompt, worker_prompt
from typing import Dict, Any
from langgraph.types import Send
from pathlib import Path

from utils import safe_filename

#  1. Orchestration logic for the blog planning process
def generate_blog_plan(state:BlogState) -> Dict:

    # 1. Get the prompt for blog planning
    prompt = get_blog_planning_prompt(state.blog_topic)

    # 2. call the structured output model to generate the plan
    response = structured_output_model.invoke(prompt)

    # print("Raw response from model:", response)

    return {'plan': response }


#  2. intermediate function between orchestrator and workers
# Now we define the fanout function for the node
# which will create multiple worker for as per task in plan 
# after the plan generated 
def fanout(state:BlogState):
    workers = []
    for task in state.plan.tasks:
        workers.append(
            Send("worker",{
                'task':task,
                'blog_topic':state.blog_topic,
                'plan':state.plan
            })
        )

    return workers

# actual generation of each task seggregated by worker will executed by worker node
def worker(payload:dict) -> dict:

    # extracting whole payload
    task = payload['task']
    blog_topic = payload['blog_topic']
    plan = payload['plan']

    # get the prompt for worker 
    prompt = worker_prompt(task, blog_topic, plan)

    # generate the content for the task using generation model
    response = generation_model.invoke(prompt).content
    # print(f"Worker response for task {task.id} - {task.title} : \n\n", response)
    return {'sections':[response]}

# reducer to aggregate all sections from workers into final blog
def reducer(state:BlogState):

    title = state.plan.blog_title
    blog = "\n\n".join(state.sections)

    final_blog = f"# {title}\n\n{blog}"

    file_name = safe_filename(title)
    print(f"Saving final blog to {file_name}...\n")

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(final_blog)


    return {'final_blog': final_blog}