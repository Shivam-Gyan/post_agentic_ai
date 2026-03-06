from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
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
    research_queries: Annotated[List[str], Field(description="If open-book or hybrid research mode is selected, a list of specific queries to use for retrieving information from external sources", default_factory=list)] 
    blog_kind:Literal["explainer",'tutorial','news_roundup','comparison','system_design'] = Field(description="The kind of blog the user wants to create", default="explainer")

# #  conversation Memory summarystate 
# class SummaryStructuredOutputSchema(BaseModel):
#     user_real_name: Optional[str]
#     user_goal: Optional[str]
#     audience: Optional[str]
#     constraints: Annotated[List[str], Field(description="A list of constraints provided by the user", default_factory=list)] 
#     preferences: Annotated[List[str], Field(description="A list of preferences provided by the user", default_factory=list)] 
#     decisions_made: Annotated[List[str], Field(description="A list of decisions made during the conversation", default_factory=list)]
#     open_questions: Annotated[List[str], Field(description="A list of open questions that need to be addressed", default_factory=list)]

class SummaryStructuredOutputSchema(BaseModel):
    # Explicit `= None` on every Optional field is REQUIRED for Pydantic V2.
    # Without it, the JSON schema marks these as "required", causing Groq
    # function-calling to reject the tool call with a 400 when the LLM omits them.
    user_real_name: Optional[str] = None
    user_professional_bio: Optional[str] = Field(default=None, description="User's role or expertise (e.g. AI Engineer).")
    current_topics_of_interest: List[str] = Field(default_factory=list, description="Topics discussed in chat (e.g. ML mistakes, Black Holes).")
    user_goal: Optional[str] = None
    audience: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    decisions_made: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


# LLm use this to generate feedback for refining the blog based on the user query 
# class FeedbackStructuredOutputSchema(BaseModel):
#     feedback: str = Field(description="Feedback for refining the blog based on the user query and the current version of the blog")

class FeedbackStructuredOutputSchema(BaseModel):
    target_section: Optional[str] = None
    action: str
    reason: str

    tone_delta: Optional[str] = None
    audience_delta: Optional[str] = None
    depth_adjustment: Optional[Literal["increase", "decrease", "same"]] = "same"

    seo_focus: Optional[
        Literal[
            "keyword_optimization",
            "heading_structure",
            "search_intent_alignment",
            "meta_description_improvement"
        ]
    ] = None

# refinment state to keep track of the refinement history and the current plan and evidence
class RefinementState(BaseModel):
    history: Annotated[List[str], Field(description="A list of blog posts in the refinement history", default_factory=list), operator.add]
    # in refinement subgraph their is a structuredparse node wich use user_query and final_blog to generate a feedback for blog updation 
    # and it sent to refine node which refine the blog based on the feedback and generate a new blog and update the hsitory and feedback List
    feedback: Annotated[List[FeedbackStructuredOutputSchema], Field(description="A list of feedback provided during the refinement process", default_factory=list), operator.add]


# intent mode detection structured output schema 
class IntentModeStructuredOutputSchema(BaseModel):
    mode: Literal["generate", "refine", "chat", "publish"] = Field(description="The mode of operation based on the user query, which can be 'generate' for generating a new blog, 'refine' for refining an existing blog, 'chat' for having a conversation with the user, or 'publish' for publishing the blog to an external platform.")

class BlogState(BaseModel):

    # this two below and are used for user_query and determine the mdoe user want like geernate a blog, refine previous blog or just have conversation
    user_query: str = Field(description="A detailed description of the blog topic provided by user", default="")
    mode: Literal["generate", "refine", "chat", "publish","guard"] = "generate"

    # conversatio state to keep track of the conversation history and the structured summary memory
    messages: Annotated[List[BaseMessage], Field(description="A list of messages in the conversation history",default_factory=list),add_messages]
    summary: SummaryStructuredOutputSchema = Field(
        default_factory=SummaryStructuredOutputSchema
    )


    # refine state to keep track of the refinement history and the current plan and evidence
    refinement: RefinementState = Field(description="The state of the refinement process including the refinement history and the current plan and evidence", default_factory=lambda: RefinementState(history=[], feedback=[]))


    blog_title: str = Field(description="The main topic of the blog provided by user", default="")
    blog_topic : str = Field(description="The main topic of the blog provided by user", default="")
    audience: str = Field(description="Who this blog is for." , default="")
    tone: str = Field(description="Writing tone (e.g., practical, crisp).", default="")
    require_research: bool = Field(description="Whether additional research is needed to fill in gaps in the blog description", default=False)
    research_mode : Literal["closed_book",'hybrid', "open_book"] = Field(description="If research is needed, whether to use a closed-book approach (rely on model's existing knowledge) or open-book approach (use external sources) or hybrid approach (combine both).", default="closed_book")
    research_queries: Annotated[List[str], Field(description="If open-book or hybrid research mode is selected, a list of specific queries to use for retrieving information from external sources")] = []
    blog_kind:Literal["explainer",'tutorial','news_roundup','comparison','system_design'] = Field(description="The kind of blog the user wants to create", default="explainer")

    plan : PlanSchema = Field(description="The plan for the blog including tasks", default_factory=lambda: PlanSchema(blog_title="", tasks=[]))  
    evidence: List[EvidenceSchema] = Field(description="The evidence retrieved from research to fill in the gaps in blog description", default_factory=list)  
    sections: Annotated[List[str], Field(description="A list of sections for the blog"), operator.add] = []

    final_blog : str = Field(description="The completed blog post", default="")
    publish_result: str = Field(description="Result of publishing the blog to external platform", default="")
