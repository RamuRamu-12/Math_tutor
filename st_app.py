# filepath: app_asdiv_streamlit.py

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import json
from agents import UnifiedMathAgent 
from dspy_feedback_agent import DSPyFeedbackAgent

load_dotenv()

# --------- INIT AGENTS ---------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
math_agent = UnifiedMathAgent()
feedback_agent = DSPyFeedbackAgent()

# --------- LLM-BASED GUARDRAILS ---------
def llm_validate_math_question(query: str) -> bool:
    validation_prompt = PromptTemplate.from_template(
        """
        You are an AI tutor. Determine if the following question is related to mathematics.
        Respond with only "yes" or "no".

        Question: {question}
        """
    )
    response = llm.invoke(validation_prompt.format(question=query)).content.strip().lower()
    return response.startswith("yes")

def llm_generate_step_by_step(query: str, final_answer: str) -> str:
    step_prompt = PromptTemplate.from_template(
        """
        You are a math teacher. Given the following question and its correct answer, provide a detailed step-by-step explanation that a student can follow.

        Question: {question}
        Final Answer: {answer}

        Step-by-step solution:
        """
    )
    return llm.invoke(step_prompt.format(question=query, answer=final_answer)).content.strip()

# --------- FEEDBACK LOGGING ---------
def log_feedback(query, answer, explanation, feedback, suggestion, refined):
    feedback_data = {
        "question": query,
        "answer": answer,
        "explanation": explanation,
        "feedback": feedback,
        "suggestion": suggestion,
        "refined_answer": refined
    }
    with open("feedback_logs.jsonl", "a") as f:
        f.write(json.dumps(feedback_data) + "\n")

# --------- STREAMLIT UI ---------
st.set_page_config(page_title="Math Agent - ASDiv", page_icon="📀")
st.title(" Math Tutor")

# Initialize session state variables
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_feedback_form' not in st.session_state:
    st.session_state.show_feedback_form = False
if 'raw_answer' not in st.session_state:
    st.session_state.raw_answer = ""
if 'explanation' not in st.session_state:
    st.session_state.explanation = ""

# Reset feedback state if it's a new question
query = st.text_input("Enter your math question:", "Tom has 3 apples and buys 2 more. How many apples does he have now?")
if query != st.session_state.current_query:
    st.session_state.feedback_submitted = False
    st.session_state.show_feedback_form = False
    st.session_state.current_query = query
    st.session_state.raw_answer = ""
    st.session_state.explanation = ""

if st.button("Get Answer") and query.strip():
    with st.spinner("Validating math question..."):
        if not llm_validate_math_question(query):
            st.warning("This assistant only supports math-related questions. Please enter a valid math question.")
            st.stop()

    with st.spinner("Thinking..."):
        try:
            st.session_state.raw_answer = math_agent.run(query)
            if st.session_state.raw_answer:
                st.session_state.explanation = llm_generate_step_by_step(query, st.session_state.raw_answer)
                st.markdown("### Answer")
                st.success(st.session_state.raw_answer)
                st.caption("Source: Auto-selected tool")

                st.markdown("### Step-by-step Solution")
                st.info(st.session_state.explanation)

                # Show feedback section only if not already submitted
                if not st.session_state.feedback_submitted:
                    st.markdown("---")
                    st.subheader("Feedback")
                    st.session_state.show_feedback_form = True

            else:
                st.markdown("### Answer")
                st.error("No valid answer found. Please try rephrasing your question or check your math.")
        except Exception as e:
            st.error(f"Agent failed: {str(e)}")

# Feedback form (shown only when needed)
if st.session_state.show_feedback_form and not st.session_state.feedback_submitted:
    feedback = st.radio("Was this answer helpful?", ("Yes", "No"), key="feedback_radio")
    
    if feedback == "No":
        user_suggestion = st.text_area("What should be improved?", key="suggestion_input")
        if st.button("Submit Feedback", key="submit_feedback_btn"):
            feedback_input = f"{query}|||{st.session_state.raw_answer}|||{st.session_state.explanation}|||{user_suggestion}"
            refined = feedback_agent.run(feedback_input)
            
            st.markdown("### Refined Answer")
            st.info(refined)
            
            log_feedback(query, st.session_state.raw_answer, st.session_state.explanation, feedback, user_suggestion, refined)
            st.session_state.feedback_submitted = True
            st.session_state.show_feedback_form = False
    elif feedback == "Yes":
        if st.button("Submit Feedback", key="submit_positive_feedback"):
            log_feedback(query, st.session_state.raw_answer, st.session_state.explanation, feedback, "", "")
            st.session_state.feedback_submitted = True
            st.session_state.show_feedback_form = False

# Thank you message after feedback submission
if st.session_state.feedback_submitted:
    st.success("Thank you for your feedback!")