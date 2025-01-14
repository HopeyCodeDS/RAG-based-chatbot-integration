import asyncio
import os
from llamaapi import LlamaAPI

import traceback
from typing import Dict, List
from langchain_chroma import Chroma
import chromadb
from dotenv import load_dotenv
from get_embedding_function import get_embedding_function
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from functools import partial
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CustomLlamaAPI:
    """Wrapper for LlamaAPI to avoid nest_asyncio issues"""

    def __init__(self, api_key):
        self.api_key = api_key
        self._api = None

    def _ensure_api(self):
        """Ensure API instance exists in current context"""
        if self._api is None:
            # Create new instance without nest_asyncio
            import llamaapi.llamaapi
            # Temporarily patch the __init__ to skip nest_asyncio
            original_init = llamaapi.llamaapi.LlamaAPI.__init__
            try:
                llamaapi.llamaapi.LlamaAPI.__init__ = lambda self, api_key: setattr(self, 'api_key', api_key)
                self._api = LlamaAPI(self.api_key)
            finally:
                llamaapi.llamaapi.LlamaAPI.__init__ = original_init
        return self._api

    def run(self, *args, **kwargs):
        """Run API call using current context"""
        api = self._ensure_api()
        return api.run(*args, **kwargs)


def init_llama_api():
    """Initialize custom LlamaAPI wrapper"""
    logger.info(f"Using API Key: {'*' * (len(os.getenv('LLAMA_API_KEY')) - 4)}{os.getenv('LLAMA_API_KEY')[-4:]}")
    try:
        if not os.getenv('LLAMA_API_KEY'):
            raise ValueError("LLAMA_API_KEY not found in environment variables")
        return CustomLlamaAPI(os.getenv('LLAMA_API_KEY'))
    except Exception as e:
        logger.error(f"Error initializing LlamaAPI: {str(e)}")
        raise


# Initialize LlamaAPI
llama = init_llama_api()
# Constants
CHROMA_PATH = "chroma"
API_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Query type patterns
GENERAL_PATTERNS = [
    'hello', 'hi', 'hey', 'help', 'can you', 'what can you do',
    'who are you', 'how do you', 'thanks', 'thank you', 'bye'
]

GAME_PATTERNS = [
    'battleship', 'battleships', 'chess', 'monopoly',
    'reversi', 'tictactoe', 'tic-tac-toe'
]

PLATFORM_PATTERNS = [
    'how to', 'where', 'find', 'navigate', 'use', 'access',
    'menu', 'search', 'platform', 'website', 'guide', 'store',
    'purchase', 'buy', 'game store', 'available games',
    'list', 'account', 'create account'
]


class QueryResponse:
    """Structure for handling query responses"""

    def __init__(self, text: str, sources: List[str]):
        self.text = text
        self.sources = sources

    def format(self) -> str:
        """Format the response for output"""
        return f"Response: {self.text}\nSources: {json.dumps(self.sources)}"


def is_pattern_match(query: str, patterns: List[str]) -> bool:
    """Check if query matches any patterns"""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in patterns)


def is_general_query(query: str) -> bool:
    return is_pattern_match(query, GENERAL_PATTERNS)


def is_game_query(query: str) -> bool:
    return is_pattern_match(query, GAME_PATTERNS)


def is_platform_query(query: str) -> bool:
    result = is_pattern_match(query, PLATFORM_PATTERNS)
    logger.info(f"Query '{query}' platform match: {result}")
    return result


def get_general_response(query: str) -> QueryResponse:
    """Generate responses for general queries"""
    query_lower = query.lower().strip('?!., ')

    # Greeting response
    if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'greetings']):
        return QueryResponse(
            text="""Hello! 👋 I'm your Game Rules Assistant. I can help you with:

            1. Game Rules: Learn how to play various games
            2. Platform Navigation: Find your way around
            3. General Questions: Get help with using the platform
            
            How can I assist you today?""",
                        sources=["greeting"]
                    )

    # Help response
    if any(pattern in query_lower for pattern in ['help', 'can you', 'what can you do']):
        return QueryResponse(
            text="""I'd be happy to help! Here's what I can do:

                1. Explain game rules in detail (e.g., "How do you play Battleship?")
                2. Help navigate the platform (e.g., "How do I find a specific game?")
                3. Answer questions about game setup and gameplay
                4. Provide guidance on using the platform features
                
                What would you like to know more about?""",
                            sources=["help"]
                        )

    # Thanks response
    if 'thank' in query_lower:
        return QueryResponse(
            text="You're welcome! Feel free to ask if you need anything else. 😊",
            sources=["thanks"]
        )

    # Default response
    return QueryResponse(
        text="""Hi there! I'm your Game Rules Assistant. I can help you with:

            1. Learning game rules
            2. Finding your way around the platform
            3. Answering general questions
            
            What would you like to know about?""",
                    sources=["default"]
                )


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def call_llama_api(api_request_json: Dict) -> Dict:
    """Make API call to LlamaAPI with retry logic and timing"""
    try:
        start_time = time.time()

        # Use run_in_executor to make the synchronous API call
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llama.run(api_request_json)
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"API Response Time: {elapsed_time:.2f} seconds")
        logger.info(f"API Response Status: {response.status_code}")
        logger.info(f"API Response Content: {response.text}")

        if response.status_code != 200:
            logger.error(f"LlamaAPI error: Status {response.status_code}")
            raise Exception(f"API call failed with status {response.status_code}")

        return response.json()
    except Exception as e:
        logger.error(f"Error calling LlamaAPI: {str(e)}")
        raise

async def query_rag(query_text: str) -> str:
    """
    Main query handling function

    Args:
        query_text: The user's query text

    Returns:
        Formatted response string with text and sources
    """
    try:
        logger.info(f"Processing query: {query_text}")

        # Handle general queries
        if is_general_query(query_text) and not is_game_query(query_text):
            logger.info("Processing as general query")
            response = get_general_response(query_text)
            logger.info(f"Response: {response.text}")
            return response.format()

        # Initialize Chroma client
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        # platform_collection = client.get_collection("platform_docs")
        # all_platform_docs = platform_collection.get()
        # print("Platform docs content:", all_platform_docs)

        embedding_function = get_embedding_function()

        # Choose collection based on query type
        collection_name = "platform_docs" if is_platform_query(query_text) else "game_rules"

        # Initialize database connection
        db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

        # Search the database
        logger.info("Searching database...")
        results = db.similarity_search_with_score(query_text, k=5)

        logger.info("Search results:")
        for doc, score in results:
            logger.info(f"Score: {score}")
            logger.info(f"Content: {doc.page_content[:50]}...")  # First 50 chars
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("---")

        if not results:
            logger.warning("No relevant results found")
            return get_general_response(query_text).format()

        # Prepare context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Prepare API request
        system_content = (
            "You are a helpful platform guide providing clear navigation assistance."
            if is_platform_query(query_text)
            else "You are a helpful assistant explaining game rules clearly and concisely."
        )

        api_request_json = {
            "model": "llama3.1-70b",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a knowledgeable and friendly gaming assistant. Provide clear, direct answers without referencing any 'context' or 'documents'. Be conversational but concise."""
                },
                {
                     "role": "user",
                    "content": f"""Based on the following information, answer the user's question in a natural, conversational way:

        
                    Context:
                    {context_text}

                    ---

                    Question: {query_text}
                    
                    Remember to use direct quotes from the context in your answer and only provide information that can be found in the context above.
                    """
                }
            ],
            "max_tokens": 512,
            "stream": False,
            "temperature": 0.7
        }

        # Call API with timeout
        try:
            response = await call_llama_api(api_request_json)
            response_text = response['choices'][0]['message']['content']

            # Get sources
            sources = [
                f"{doc.metadata['source']}:page{doc.metadata['page']}"
                for doc, _score in results
            ]

            return QueryResponse(response_text, sources).format()

        except asyncio.TimeoutError:
            logger.error("API call timed out")
            raise TimeoutError("Request timed out")

        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            return QueryResponse(
                "I apologize, but I'm having trouble processing your request right now. Please try again.",
                ["error"]
            ).format()

    except Exception as e:
        logger.error(f"Error in query_rag: {str(e)}")
        logger.error(traceback.format_exc())
        return QueryResponse(
            "I'm here to help! Please feel free to ask about game rules or how to use the platform.",
            ["system_error"]
        ).format()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text to process")
    args = parser.parse_args()

    # Run the query
    asyncio.run(query_rag(args.query_text))