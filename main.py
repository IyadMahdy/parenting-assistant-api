from fastapi import FastAPI
from routers import chat, users

app = FastAPI(title="Parenting Assistant API")

# Register routers
app.include_router(users.router, prefix="/api", tags=["Users"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])

@app.get("/")
def root():
    return {"message": "Parenting Assistant API is running!"}