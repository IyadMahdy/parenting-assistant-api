from fastapi import APIRouter
from models.schemas import ChatRequest, ChatResponse
from services.chains import conversational_rag_chain, get_session_history
from routers.users import user_store

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Personalize with user data
    user_info = user_store.get(req.user_id)
    user_context = user_info.dict() if user_info else "No user info provided."

    # Run chain
    response = conversational_rag_chain.invoke(
        {
            "input": req.message,
            "user_info": user_context
        },
        config={"configurable": {"session_id": req.user_id}}
    )
    answer = response["answer"]

    # Update memory
    history = get_session_history(req.user_id)
    history.add_user_message(req.message)
    history.add_ai_message(answer)

    return ChatResponse(answer=answer)