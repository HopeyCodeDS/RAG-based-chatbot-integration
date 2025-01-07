from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import ChatMessage, ChatResponse
import traceback
from query_data import query_rag

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Platform Chatbot API"}

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    try:
        # Print received message for debugging
        print(f"Received message: {message.message}")

        # Call query_rag and get response
        response_text = query_rag(message.message)
        print(f"Raw response: {response_text}")

        # Parse the response
        parts = response_text.split("\nSources: ")
        main_response = parts[0].replace("Response: ", "").strip()
        sources = eval(parts[1]) if len(parts) > 1 else []

        # Return clean response
        return {
            "response": main_response,
            "sources": sources
        }

    except Exception as e:
        print("Error occurred:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
