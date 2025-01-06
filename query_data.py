import argparse
from http.client import responses

from llamaapi import LlamaAPI
from langchain_chroma import Chroma
import chromadb
import os
from dotenv import load_dotenv

from get_embedding_function import get_embedding_function

# Load environment variables
load_dotenv()

# Initialize LlamaAPI
llama = LlamaAPI(os.getenv('LLAMA_API_KEY'))

CHROMA_PATH = "chroma"


def is_general_query(query: str) -> bool:
    general_patterns = [
        'hello', 'hi', 'hey', 'help', 'can you', 'what can you do?',
        'who are you', 'how do you', 'thanks', 'thank you', 'bye'
    ]
    return any(pattern in query.lower() for pattern in general_patterns)

def is_platform_query(query: str) -> bool:
    platform_patterns = [
        'how to', 'where', 'find', 'navigate', 'use', 'access',
        'menu', 'search', 'platform', 'website', 'guide'
    ]
    return any(pattern in query.lower() for pattern in platform_patterns)

def get_general_response(query: str) -> str:
    query_lower = query.lower()

    # Greetings
    if any(word in query_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your Game Rules Assistant. I can help you with game rules, platform navigation, and general questions. How can I assist you today?"

    # Help requests
    if any(word in query_lower for word in ['help', 'can you', 'what can you do']):
        return """I can help you with:
            1. Game Rules: Learn how to play various games like Chess, Monopoly, and more
            2. Platform Navigation: Find your way around the platform
            3. General Questions: Get assistance with using the chatbot and finding information
            
            What would you like to know about?"""

    # Thanks
    if 'thank' in query_lower:
        return "You're welcome! Let me know if you need anything else."

    # Default response
    return "I'm here to help with game rules and platform navigation. Could you please ask me about specific games or how to use the platform?"



def query_rag(query_text: str):
    try:
        # Check if it's a general query first
        if is_general_query(query_text):
            response_text = get_general_response(query_text)
            return f"Response: {response_text}\nSources: [conversational]"

        # Initialize Chroma with a specific collection name
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_function = get_embedding_function()

        # Choose collection based on query type
        collection_name = "platform_docs" if is_platform_query(query_text) else "game_rules"

        db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

        # Search the DB
        results = db.similarity_search_with_score(query_text, k=5)

        # Prepare context from search results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Adjust prompt based on query type
        if is_platform_query(query_text):
            system_content = "You are a helpful platform guide providing clear navigation assistance."
        else:
            system_content = "You are a helpful assistant explaining game rules clearly and concisely."

        # Construct API request
        api_request_json = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system",
                 "content": system_content},
                {"role": "user", "content": f"""
                            Answer the question based only on the following context:
        
                            Context:
                            {context_text}
        
                            ---
        
                            Question: {query_text}
                            """
                },
            ],
            "max_tokens": 512,
            "stream": False
        }

        # Call API
        response = llama.run(api_request_json)
        response_text = response.json()['choices'][0]['message']['content']

        # Get sources and format response
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return formatted_response

    except Exception as e:
        print(f"Error in query_rag: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


if __name__ == "__main__":
    main()
