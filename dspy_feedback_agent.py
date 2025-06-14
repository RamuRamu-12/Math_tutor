# filepath: dspy_feedback_agent.py

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFinetune  # FIXED



# DSPy Signature for feedback refinement
class FeedbackReview(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.InputField()
    explanation = dspy.InputField()
    feedback = dspy.InputField()
    refined_answer = dspy.OutputField(desc="Improved version of the answer or 'No change needed'.")

# DSPy Module wrapping the feedback signature
class DSPyFeedbackReviewer(dspy.Module):
    def __init__(self):
        super().__init__()  # FIXED: include super init
        self.predict = dspy.Predict(FeedbackReview)

    def forward(self, question, answer, explanation, feedback):
        return self.predict(
            question=question,
            answer=answer,
            explanation=explanation,
            feedback=feedback
        )

# Configure DSPy with GPT-4
lm = dspy.LM(model="gpt-4o-mini", max_tokens=512)
dspy.settings.configure(lm=lm)

# Feedback reviewer instance (can be tuned)
reviewer = DSPyFeedbackReviewer()

# Example: Load trainset from feedback logs (JSONL format)
def load_trainset(filepath="feedback_logs.jsonl"):
    import json
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f.readlines()]

# DSPy tuning wrapper
def tune_feedback_agent(trainset_path="feedback_logs.jsonl"):
    dataset = load_trainset(trainset_path)
    tuner = BootstrapFinetune(metric="exact_match")  # FIXED
    reviewer.predict = tuner.compile(
        reviewer.predict,
        trainset=dataset,
    )
    return reviewer

# LangChain tool using DSPy reviewer
def refine_answer_with_feedback(input_str: str) -> str:
    try:
        question, answer, explanation, feedback = input_str.split("|||")
        result = reviewer.forward(
            question=question.strip(),
            answer=answer.strip(),
            explanation=explanation.strip(),
            feedback=feedback.strip()
        )
        return result.refined_answer
    except Exception as e:
        return f"Invalid input format or error: {str(e)}"

def DSPyFeedbackAgent():
    feedback_tools = [
        Tool(
            name="refine_answer_with_feedback",
            func=refine_answer_with_feedback,
            description="Refines a math answer based on human feedback. Input: <question>|||<answer>|||<explanation>|||<feedback>"
        )
    ]

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    DSPyAgent = initialize_agent(
        tools=feedback_tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return DSPyAgent

# CLI demo
if __name__ == "__main__":
    # Optional: tune on historical feedback
    # tune_feedback_agent("feedback_logs.jsonl")

    prompt = """
    Tom has 3 apples and buys 2 more.|||Tom has 5 apples now.|||To solve this, add 3 and 2 together.|||The explanation should be more detailed.
    """
    agent = DSPyFeedbackAgent()
    result = agent.run(f"refine_answer_with_feedback: {prompt.strip()}")
    print("\nFinal Output:")
    print(result)
