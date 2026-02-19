from pydantic import BaseModel, Field
from typing import Literal, Annotated,List,Optional
import operator


# Improved TaskSchema with operator annotations for merging tasks
class TaskSchema(BaseModel):
    id: int
    title: str
    goal: str = Field(description="One sentence describing what the reader should be able to do/understand after this section.",)
    bullets: List[str] = Field(min_length=3,max_length=5, description="3–5 concrete, non-overlapping subpoints to cover in this section.",)
    target_words: int = Field(description="Target word count for this section (120–450).",)
    section_type: Literal[ "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field( description="Use 'common_mistakes' exactly once in the plan.",)
    require_research: bool = Field(description="Whether this task requires additional research to fill in gaps in the blog description", default=False)
    require_citations: bool = Field(description="Whether this task requires citations for any claims made", default=False)
    require_code: bool = Field(description="Whether this task requires code snippets or technical explanations", default=False)


class PlanSchema(BaseModel):
    blog_title: str = Field(description="The title of the plan")
    tasks: Annotated[List[TaskSchema], Field(description="A list of tasks included in the plan")]
    
    


class EvidenceSchema(BaseModel):
    title: str = Field(description="Title of the evidence")
    content: Optional[str] = Field(description="Content of the evidence, if applicable")
    url: str = Field(description="URL of the evidence")
    # source: Optional[str] = Field(description="Source of the evidence, if applicable")
    # published_date: Optional[str] = Field(description="Published date of the evidence, if applicable")


class EvidencePackSchema(BaseModel):
    evidence: List[EvidenceSchema] = Field(description="A list of evidence items related to the research queries")



class ResearchSchema(BaseModel):
    topic: str = Field(description="The main topic of the blog provided by user")
    description: str = Field(description="A detailed description of the blog topic provided by user")
    audience: str = Field(description="Who this blog is for.")
    tone: str = Field(description="Writing tone (e.g., practical, crisp).")

    require_research: bool = Field(description="Whether additional research is needed to fill in gaps in the blog description", default=False)
    research_mode : Literal["closed_book",'hybrid', "open_book"] = Field(description="If research is needed, whether to use a closed-book approach (rely on model's existing knowledge) or open-book approach (use external sources) or hybrid approach (combine both).", default="closed_book")
    research_queries: Annotated[List[str], Field(description="If open-book or hybrid research mode is selected, a list of specific queries to use for retrieving information from external sources")] = []
    blog_kind:Literal["explainer",'tutorial','news_roundup','comparison','system_design'] = Field(description="The kind of blog the user wants to create", default="explainer")


class BlogState(BaseModel):
    blog_description: str = Field(description="A detailed description of the blog topic provided by user", default="")

    blog_title: str = Field(description="The main topic of the blog provided by user", default="")
    blog_topic : str = Field(description="The main topic of the blog provided by user", default="")
    audience: str = Field(description="Who this blog is for." , default="")
    tone: str = Field(description="Writing tone (e.g., practical, crisp).", default="")
    require_research: bool = Field(description="Whether additional research is needed to fill in gaps in the blog description", default=False)
    research_mode : Literal["closed_book",'hybrid', "open_book"] = Field(description="If research is needed, whether to use a closed-book approach (rely on model's existing knowledge) or open-book approach (use external sources) or hybrid approach (combine both).", default="closed_book")
    research_queries: Annotated[List[str], Field(description="If open-book or hybrid research mode is selected, a list of specific queries to use for retrieving information from external sources")] = []
    blog_kind:Literal["explainer",'tutorial','news_roundup','comparison','system_design'] = Field(description="The kind of blog the user wants to create", default="explainer")

    plan : PlanSchema = Field(description="The plan for the blog including tasks", default_factory=lambda: PlanSchema(blog_title="", tasks=[]))  
    evidence: List[EvidenceSchema] = Field(description="The evidence retrieved from research to fill in the gaps in blog description", default=[])  
    sections: Annotated[List[str], Field(description="A list of sections for the blog"), operator.add] = []

    final_blog : str = Field(description="The completed blog post", default="")
    publish_result: str = Field(description="Result of publishing the blog to external platform", default="")
