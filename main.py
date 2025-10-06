from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot import get_bot_response
from app.models import ChatRequest, ChatResponse

app = FastAPI(
    title="Chatbot API",
    description="A simple API for the LangChain chatbot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://roc-frontend.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Receives a question and session ID, and returns the chatbot's response.
    """
    response = get_bot_response(request.question, request.session_id)
    return ChatResponse(response=response)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API"}