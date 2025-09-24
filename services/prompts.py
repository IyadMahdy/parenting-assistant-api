from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_system_prompt = """
Rephrase the user question into a standalone form if it depends on chat history. 
Do not answer, only return the rephrased or original question.
""".strip()

contextualize_prompt_template = ChatPromptTemplate(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

system_prompt = """
You are a parenting assistant.
Answer using the retrieved context and user-specific information if available.
If no context helps, reply with your knowledge or say "I don't know."
Be concise (<200 words), friendly, and empathetic.
Advise professional help if needed.

Context:
{context}

User Info (if provided):
{user_info}
""".strip()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)