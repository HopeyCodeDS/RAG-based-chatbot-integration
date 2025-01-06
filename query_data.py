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

def is_game_query(query: str) -> bool:
    game_names = [
        'battleship', 'battleships', 'chess', 'monopoly',
        'reversi', 'tictactoe', 'tic-tac-toe'
    ]
    query_lower = query.lower()
    return any(game in query_lower for game in game_names)

def is_platform_query(query: str) -> bool:
    platform_patterns = [
        'how to', 'where', 'find', 'navigate', 'use', 'access',
        'menu', 'search', 'platform', 'website', 'guide'
    ]
    return any(pattern in query.lower() for pattern in platform_patterns)


def get_general_response(query: str) -> str:
    query_lower = query.lower().strip('?!., ')

    # Greeting patterns
    greetings = ['hello', 'hi', 'hey', 'greetings']
    if any(greeting in query_lower for greeting in greetings):
        return """Hello! 👋 I'm your Game Rules Assistant. I can help you with:

1. Game Rules: Learn how to play various games
2. Platform Navigation: Find your way around
3. General Questions: Get help with using the platform

How can I assist you today?"""

    # Help patterns
    help_patterns = ['help', 'can you', 'what can you do']
    if any(pattern in query_lower for pattern in help_patterns):
        return """I'd be happy to help! Here's what I can do:

1. Explain game rules in detail (e.g., "How do you play Battleship?")
2. Help navigate the platform (e.g., "How do I find a specific game?")
3. Answer questions about game setup and gameplay
4. Provide guidance on using the platform features

What would you like to know more about?"""

    # Thanks patterns
    if 'thank' in query_lower:
        return "You're welcome! Feel free to ask if you need anything else. 😊"

    # Default response
    return """Hi there! I'm your Game Rules Assistant. I can help you with:

1. Learning game rules
2. Finding your way around the platform
3. Answering general questions

What would you like to know about?"""



def query_rag(query_text: str):
    try:
        # Check if it's a general query first
        if is_general_query(query_text):
            response_text = get_general_response(query_text)
            formatted_response = f"Response: {response_text}\nSources: [conversational]"
            print(formatted_response)
            return formatted_response

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
        print("\n🔎 Searching database...")
        results = db.similarity_search_with_score(query_text, k=5)

        # Debug: Print search results
        print("\n📄 Search Results:")
        for doc, score in results:
            print(f"\nDocument: {doc.metadata.get('source')}")
            print(f"Score: {score}")
            print(f"Content Preview: {doc.page_content[:150]}...")

        # Check if we got any relevant results
        if not results:
            print("\n❌ No relevant results found")
            return get_general_response(query_text)

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
        error_msg = f"\n❌ Error in query_rag: {str(e)}"
        print(error_msg)

        # Print full error details
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())

        # Fallback response for errors
        fallback_response = "I'm here to help! Please feel free to ask about game rules or how to use the platform."
        formatted_response = f"Response: {fallback_response}\nSources: [system]"
        print(formatted_response)
        return formatted_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


if __name__ == "__main__":
    main()
