# filepath: unified_math_agent.py

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.tools.tavily_search import TavilySearchResults
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configs
QDRANT_COLLECTION = "asdiv_math"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# ---- Tool 1: ASDiv Retrieval ----
def retrieve_asdiv_answer(query: str) -> str:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa_chain.invoke({"query": query})
    return result["result"]

asdiv_tool = Tool(
    name="asdiv_math_qa",
    func=retrieve_asdiv_answer,
    description="Use this to answer math-related questions using the ASDiv dataset. Best for school-style math word problems."
)

# ---- Tool 2: Web Search Fallback ----
search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

def perform_math_websearch_fallback(query: str) -> str:
    results = search_tool.invoke({"query": query})
    if not results:
        return "No relevant web content found."
    context = "\n\n".join([r["content"] for r in results[:2]])
    prompt = PromptTemplate.from_template("""
        Answer the following math question using only the context provided.

        Context:
        {context}

        Question: {question}
        """)
    return llm.invoke(prompt.format(context=context, question=query)).content

web_tool = Tool(
    name="math_web_search_fallback",
    func=perform_math_websearch_fallback,
    description="Use this when the ASDiv dataset does not contain a suitable answer. Answers math questions using recent web results."
)

# ---- Unified Agent with Both Tools ----
def UnifiedMathAgent():
    tools = [asdiv_tool, web_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,  # ensures only valid tool names used
        verbose=True
    )
    return agent

# ---- CLI test ----
if __name__ == "__main__":
    agent = UnifiedMathAgent()
    question = "If Tom has 1720 apples and shares them with 43 students, how many apples per student?"
    print("Question:", question)
    answer = agent.run(question)
    print("Answer:", answer)
