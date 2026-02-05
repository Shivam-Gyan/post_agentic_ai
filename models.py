from langchain_ollama import ChatOllama
from states import PlanSchema


#  1. model used to generate a structured output according to the plan Schema
structure_model = ChatOllama(model='qwen3:1.7b',temperature=0.4) 
structured_output_model = structure_model.with_structured_output(PlanSchema)


# 2. model used to generate the blog sections and final blog post
generation_model = ChatOllama(model='ministral-3:3b',temperature=0.4)