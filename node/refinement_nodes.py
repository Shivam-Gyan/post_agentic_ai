


from prompts.refinement_prompts import get_feedback_prompt, get_refine_blog_prompt
from states import BlogState, FeedbackStructuredOutputSchema
from models import refine_feedback_output_model, get_generation_model

async def refine_structured_output_model(state: BlogState) -> dict:

    try:
        # get the previous feedback if exists else default message
        prev_feedback = state.refinement.feedback[-1] if state.refinement.feedback else FeedbackStructuredOutputSchema(action="No feedback provided", reason="No feedback provided") # type: ignore
        # generate the prompt for refinement feedback
        refine_feedabck_prompt = get_feedback_prompt(state.user_query, state.final_blog, prev_feedback)

        # invoke the model to get the refinement feedback
        response = await refine_feedback_output_model.ainvoke(refine_feedabck_prompt) 
        print(f"Refinement feedback response: {response}") 
        # print(f"\nRefinement feedback generated successfully")
        return {
            "refinement": {
                "feedback": [response] # type: ignore
            }
        }
    except Exception as e:
        print(f"Error in refine_structured_output_model: {e}")
        raise e



async def refine_node_func(state: BlogState)->dict:


    try:

        refined_blog_history = state.refinement.history[-1] if state.refinement.history else "No previous blog to refine"
        feedback_to_refine_blog = state.refinement.feedback[-1] if state.refinement.feedback else FeedbackStructuredOutputSchema(action="No feedback provided", reason="No feedback provided") # type: ignore

        user_query = state.user_query
        blog = state.final_blog if state.final_blog else "No blog generated yet to refine"
        blog_title = state.blog_title if state.blog_title else "No blog title provided"
        blog_plan = state.plan if state.plan else None

        # extract only task titles from the plan
        plan_titles = [task.title for task in blog_plan.tasks] if blog_plan and blog_plan.tasks else []

        # build the refinement prompt
        refine_prompt = get_refine_blog_prompt(
            user_query=user_query,
            blog=blog,
            feedback=feedback_to_refine_blog,
            refined_blog_history=refined_blog_history,
            blog_title=blog_title,
            blog_plan_titles=plan_titles
        )

        # invoke the generation model to produce the refined blog
        generation_model = get_generation_model()
        response = await generation_model.ainvoke(refine_prompt)
        refined_blog = response.content

        print(f"\nRefined blog generated successfully")
        return {
            "final_blog": refined_blog,
            "refinement": {
                "history": [refined_blog],
            },
        }
    except Exception as e:
        print(f"Error in refine_node_func: {e}")
        raise e