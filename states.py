from pydantic import BaseModel, Field
from typing import Literal, Annotated,List
import operator


# Basic schema for a task in the plan
# class TaskSchema(BaseModel):
#     id: int
#     title: str = Field(description="The title of the task",default="")
#     description: str = Field(description="A brief description of the task",default="")


# Improved TaskSchema with operator annotations for merging tasks
class TaskSchema(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120–450).",
    )
    section_type: Literal[
        "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )


class PlanSchema(BaseModel):
    blog_title: str = Field(description="The title of the plan")
    tasks: Annotated[List[TaskSchema], Field(description="A list of tasks included in the plan")]



class BlogState(BaseModel):
    blog_description: str = Field(description="A detailed description of the blog topic provided by user")

    blog_topic : str = Field(description="The main topic of the blog provided by user", default="")
    audience: str = Field(description="Who this blog is for." , default="")
    tone: str = Field(description="Writing tone (e.g., practical, crisp).", default="")
    plan : PlanSchema = Field(description="The plan for the blog including tasks", default = PlanSchema(blog_title="", tasks=[]))  

    sections: Annotated[List[str], Field(description="A list of sections for the blog"), operator.add] = []

    final_blog : str = Field(description="The completed blog post", default="")


class DetailsSchema(BaseModel):
    topic: str = Field(description="The main topic of the blog provided by user")
    description: str = Field(description="A detailed description of the blog topic provided by user")
    audience: str = Field(description="Who this blog is for.")
    tone: str = Field(description="Writing tone (e.g., practical, crisp).")