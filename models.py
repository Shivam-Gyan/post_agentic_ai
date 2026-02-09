from langchain_ollama import ChatOllama
from states import DetailsSchema, PlanSchema
import itertools
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#  1. model used to generate a structured output according to the plan Schema
# structure_model = ChatOllama(model='qwen3:1.7b',temperature=0.4) 
structure_model = ChatGroq(
    model='qwen/qwen3-32b',
    api_key = GROQ_API_KEY, # type: ignore
    temperature=0.4
) 

# model to extract details from the initial blog description provided by user
detail_structured_output_model = structure_model.with_structured_output(DetailsSchema) 

 # planning model to generate the plan for the blog
structured_output_model = structure_model.with_structured_output(PlanSchema)


# 2. model used to generate the blog sections and final blog post

# GEN_MODELS = [
#     ChatOllama(model='ministral-3:3b',temperature=0.4),
#     ChatOllama(model='qwen3:1.7b-q4_K_M',temperature=0.4) 
# ]


# _generation_cycle = itertools.cycle(GEN_MODELS)

# generation_model = ChatOllama(model='deepseek-r1:1.5b',temperature=0.4)

generation_model = ChatGroq(
    #  model="llama-3.3-70b-versatile",
    #  model="openai/gpt-oss-120b",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=GROQ_API_KEY, #type: ignore
    temperature=0.4
    )

def get_generation_model():
    return generation_model