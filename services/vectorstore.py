from dotenv import dotenv_values
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env
env = dotenv_values("app.env")
QDRANT_API_KEY = env.get("QDRANT_API_KEY")

COLLECTION_NAME = "parenting-tips"


def load_vectorstore() -> QdrantVectorStore:
    """Load the Qdrant vectorstore for use in retrievers/chains."""

    client = QdrantClient(
        url="https://11aa67b8-969b-4809-b921-64b1cbf0f91a.eu-central-1-0.aws.cloud.qdrant.io",
        api_key=QDRANT_API_KEY,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    return vectorstore
