from langchain_ollama import ChatOllama
from states import EvidencePackSchema, IntentModeStructuredOutputSchema, ResearchSchema, PlanSchema, FeedbackStructuredOutputSchema, SummaryStructuredOutputSchema
import itertools
from langchain_groq import ChatGroq
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#  1. model used to generate a structured output according to the plan Schema
# structure_model = ChatOllama(model='qwen3:1.7b',temperature=0.4) 
structure_model = ChatGroq(
    model='qwen/qwen3-32b',
    # model='openai/gpt-oss-120b',
    api_key = GROQ_API_KEY, # type: ignore
    temperature = 0 # Critical for schema adherence
) 
# structure_model = ChatOllama(
#     model='qwen3:1.7b-q4_K_M',
#     temperature = 0 # Critical for schema adherence
# ) 

# intent mode detetcion model

conversation_summary_structured_output_model = structure_model.with_structured_output(SummaryStructuredOutputSchema, method="function_calling")

# refine feedback structured output model
refine_feedback_output_model = structure_model.with_structured_output(FeedbackStructuredOutputSchema, method="function_calling")

# model to extract details from the initial blog description provided by user
research_structured_output_model = structure_model.with_structured_output(ResearchSchema, method="function_calling") 

 # planning model to generate the plan for the blog
structured_output_model = structure_model.with_structured_output(PlanSchema, method="function_calling")

# model to perform research based on the research queries provided by the orchestrator node
structured_output_model_research = structure_model.with_structured_output(EvidencePackSchema, method="function_calling")


generation_model = ChatGroq(
    #  model="llama-3.3-70b-versatile",
    #  model="openai/gpt-oss-120b",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=GROQ_API_KEY, #type: ignore
    temperature=0.9,   # ← higher = more varied retries
    model_kwargs={
        "top_p": 0.95,  # ← also helps with diversity
    }
)
# generation_model = ChatOllama(
#     #  model="llama-3.3-70b-versatile",
#     #  model="openai/gpt-oss-120b",
#     model="deepseek-r1:1.5b", #type: ignore
#     # api_key=GROQ_API_KEY, #type: ignore
#     temperature=0.4
# )

def get_generation_model():
    return generation_model





#  text to speech model using groq
text_to_speech_model = Groq(api_key=GROQ_API_KEY)