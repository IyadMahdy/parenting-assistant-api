from pydantic import BaseModel
from typing import List, Optional

# --- Children ---
class Child(BaseModel):
    id: int
    name: str
    male: bool
    dateOfBirth: str
    medicalNotes: Optional[str] = None
    medicalRecords: List[dict] = []
    growthRecords: List[dict] = []
    diseasesAndAllergies: List[str] = []
    meals: List[dict] = []

# --- User (Parent) ---
class User(BaseModel):
    id: int
    name: str
    phone: str
    email: str
    male: bool
    familyId: str
    children: List[Child] = []

# --- Chat Request/Response ---
class ChatRequest(BaseModel):
    user_id: int
    message: str

class ChatResponse(BaseModel):
    answer: str