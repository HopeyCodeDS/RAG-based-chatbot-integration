from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import ChatMessage, ChatResponse
import traceback
from query_data import query_rag
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load query_rag
query_rag = None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    try:
        # Lazy load imports
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
@app.get("/")
async def read_root():
    return {"message": "Platform Chatbot API"}

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    try:
        global query_rag
        if query_rag is None:
            logger.info("Lazy loading query_rag...")
            from query_data import query_rag

        # Print received message for debugging
        logger.info(f"Received message: {message.message}")

        # Call query_rag and get response
        response_text = query_rag(message.message)
        logger.info(f"Raw response: {response_text}")

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
        logger.error("Error occurred:")
        logger.info(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

