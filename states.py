from pydantic import BaseModel, Field
from typing import Literal, Annotated,List
import operator

class TaskSchema(BaseModel):
    id: int
    title: str = Field(description="The title of the task",default="")
    description: str = Field(description="A brief description of the task",default="")


class PlanSchema(BaseModel):
    blog_title: str = Field(description="The title of the plan",default="")
    tasks: Annotated[List[TaskSchema], Field(description="A list of tasks included in the plan")] = []

class BlogState(BaseModel):
    blog_topic : str = Field(description="The main topic of the blog provided by user",default="")
    plan : PlanSchema = Field(description="The plan for the blog including tasks",default=PlanSchema())

    sections: Annotated[List[str], Field(description="A list of sections for the blog"), operator.add] = []

    final_blog : str = Field(description="The completed blog post",default="")