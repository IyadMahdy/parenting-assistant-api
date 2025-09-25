import os
from dotenv import dotenv_values
from langchain_groq import ChatGroq

# env = dotenv_values("app.env")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
HF_TOKEN  = os.getenv("HF_TOKEN")

def load_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=256,
        reasoning_effort="none"
    )