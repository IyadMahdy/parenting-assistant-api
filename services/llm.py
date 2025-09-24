import os
from dotenv import dotenv_values
from langchain_groq import ChatGroq

env = dotenv_values("app.env")
os.environ["GROQ_API_KEY"] = env["GROQ_API_KEY"]
os.environ["HF_TOKEN"] = env["HF_TOKEN"]

def load_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.3,
        max_tokens=256,
        reasoning_effort="none"
    )