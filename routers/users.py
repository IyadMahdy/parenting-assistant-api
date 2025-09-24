from fastapi import APIRouter
from models.schemas import User

router = APIRouter()
user_store = {}

@router.post("/users")
def add_user(user: User):
    user_store[user.id] = user
    return {"message": f"User {user.name} stored successfully."}

@router.get("/users/{user_id}")
def get_user(user_id: int):
    return user_store.get(user_id, {"error": "User not found"})