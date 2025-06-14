from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import os

# Load env vars
load_dotenv()

# Config
ASDIV_XML_PATH = "D:\PERSONAL_PROJECTS\Genai_training_rag\AI_PLANET\dataset\ASDiv.xml"
QDRANT_COLLECTION = "asdiv_math"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def load_asdiv_to_qdrant(xml_path: str = ASDIV_XML_PATH) -> str:
    """Parse and upload ASDiv XML to Qdrant vector store"""
    documents = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return f"[!] XML parse error: {e}"

    problems = root.findall(".//Problem")

    for idx, problem in enumerate(problems):
        try:
            pid = problem.findtext("ID", default="").strip()
            body = problem.findtext("Body", default="").strip()
            question = problem.findtext("Question", default="").strip()
            formula = problem.findtext("Formula", default="").strip()
            answer = problem.findtext("Answer", default="").strip()
            full_question = f"{body} {question}".strip()

            if full_question:
                documents.append(Document(
                    page_content=full_question,
                    metadata={"id": pid, "equation": formula, "answer": answer}
                ))
        except Exception as e:
            continue

    if not documents:
        return "[!] No valid problems found."

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(size=1536, distance=qdrant_models.Distance.COSINE)
        )
    except Exception as e:
        return f"[!] Qdrant setup error: {e}"

    embeddings = OpenAIEmbeddings()
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    Qdrant.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION
    )

    return f"[✓] Uploaded {len(documents)} ASDiv entries to Qdrant."

def ASDivStorageAgent():
    """Agent dedicated to loading and storing the ASDiv dataset"""
    tools = [
        Tool(
            name="load_asdiv_math_dataset",
            func=load_asdiv_to_qdrant,
            description="Use this to upload the ASDiv math dataset to the Qdrant vector store."
        )
    ]

    agent = initialize_agent(
        llm=ChatOpenAI(temperature=0),
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

if __name__ == "__main__":
    agent = ASDivStorageAgent()
    print(agent.run("Load the ASDiv dataset into the vector store"))