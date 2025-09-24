from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from services.llm import load_llm
from services.vectorstore import load_vectorstore
from services.prompts import contextualize_prompt_template, prompt_template

llm = load_llm()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Retriever aware of history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt_template
)

# QA chain
history_question_answering_chain = create_stuff_documents_chain(
    llm, prompt_template
)

# Retrieval + QA
history_rag_chain = create_retrieval_chain(
    history_aware_retriever, history_question_answering_chain
)

# --- Memory ---
store = {}

def get_trimmed_history(user_id: str, keep_pairs: int = 3):
    if user_id not in store:
        store[user_id] = ChatMessageHistory()
    history = store[user_id]

    if len(history.messages) > keep_pairs * 2:
        history.messages = history.messages[-keep_pairs * 2 :]
    return history

def get_session_history(user_id: str) -> BaseChatMessageHistory:
    return get_trimmed_history(user_id, keep_pairs=3)

# Main chain
conversational_rag_chain = RunnableWithMessageHistory(
    history_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)