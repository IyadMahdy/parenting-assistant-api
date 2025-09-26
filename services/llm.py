import os
from dotenv import dotenv_values
from langchain_groq import ChatGroq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one folder up from services/
env_path = os.path.join(BASE_DIR, "app.env")

env = dotenv_values(env_path)
GROQ_API_KEY = env.get("GROQ_API_KEY")

print("Loaded env:", env)
print("GROQ_API_KEY:", GROQ_API_KEY)

def load_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=256,
        reasoning_effort="none"
    )