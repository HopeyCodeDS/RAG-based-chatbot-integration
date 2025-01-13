from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import ChatMessage, ChatResponse
import traceback
import logging
from datetime import datetime

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Platform Chatbot API",
    description="API for Platform Chatbot",
    version="1.0.0"
)

# CORS configuration - restrict to specific domains
origins = [
    "https://mango-sky-053dae803.4.azurestaticapps.net",
    "http://localhost:3000",  # Development environment
    "http://localhost:5173",  # Vite development server
    "http://localhost:8060"  # Local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load query_rag
query_rag = None


@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup"""
    logger.info("Starting application...")
    try:
        # Add any startup initialization here
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.get("/")
async def read_root():
    """Root endpoint for API health check"""
    return {
        "message": "Platform Chatbot API",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """
    Process chat messages and return responses
    """
    try:
        global query_rag
        if query_rag is None:
            logger.info("Lazy loading query_rag...")
            from query_data import query_rag

        # Log received message
        logger.info(f"Received message: {message.message}")

        # Call query_rag and await its response
        response_text = await query_rag(message.message)
        logger.info(f"Raw response: {response_text}")

        # Parse the response
        parts = response_text.split("\nSources: ")
        main_response = parts[0].replace("Response: ", "").strip()

        # Safely parse sources
        sources = []
        if len(parts) > 1:
            try:
                sources_str = parts[1].strip("[]").replace("'", "").replace('"', "")
                sources = [s.strip() for s in sources_str.split(",") if s.strip()]
            except Exception as e:
                logger.warning(f"Error parsing sources: {str(e)}")

        # Return properly formatted response
        return ChatResponse(
            response=main_response,
            sources=sources
        )

    except Exception as e:
        logger.error("Error occurred:")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }