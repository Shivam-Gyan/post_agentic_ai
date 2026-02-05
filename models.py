from langchain_ollama import ChatOllama
from states import DetailsSchema, PlanSchema
import itertools


#  1. model used to generate a structured output according to the plan Schema
structure_model = ChatOllama(model='qwen3:1.7b',temperature=0.4) 
structured_output_model = structure_model.with_structured_output(PlanSchema) # planning model to generate the plan for the blog
detail_structured_output_model = structure_model.with_structured_output(DetailsSchema) # model to extract details from the initial blog description provided by user


# 2. model used to generate the blog sections and final blog post

# GEN_MODELS = [
#     ChatOllama(model='ministral-3:3b',temperature=0.4),
#     ChatOllama(model='qwen3:1.7b-q4_K_M',temperature=0.4) 
# ]


# _generation_cycle = itertools.cycle(GEN_MODELS)

def get_generation_model():
    return ChatOllama(model='deepseek-r1:1.5b',temperature=0.4)