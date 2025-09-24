from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    return FAISS.load_local(
        "data/parenting_faiss_qwen3",
        embeddings,
        allow_dangerous_deserialization=True
    )