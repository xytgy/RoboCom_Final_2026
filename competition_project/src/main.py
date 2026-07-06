from fastapi import FastAPI
from pydantic import BaseModel
from model_service import generate_response

class ChatRequest(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    response = generate_response(request.message)
    return {"response": response}