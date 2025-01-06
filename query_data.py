import argparse
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


def determine_query_type(query: str) -> str:
    # Keywords that suggest platform-related queries
    platform_keywords = ['platform', 'website', 'login', 'account', 'how to use', 'navigation']

    # Keywords that suggest FAQ queries
    faq_keywords = ['faq', 'frequently', 'common', 'issue', 'problem', 'help']

    query_lower = query.lower()

    if any(keyword in query_lower for keyword in platform_keywords):
        return "platform_docs"
    elif any(keyword in query_lower for keyword in faq_keywords):
        return "faqs"
    else:
        return "game_rules"

def query_rag(query_text: str):
    try:
        # Initialize Chroma with a specific collection name
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_function = get_embedding_function()

        # Determine which collection to query
        collection_name = determine_query_type(query_text)

        db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

        # Search the DB
        results = db.similarity_search_with_score(query_text, k=5)

        # Prepare context from search results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Select prompt based on collection type
        if collection_name == "game_rules":
            system_prompt = """You are a game rules expert. Present the rules in a clear, structured format:
                        1. Start with game objective
                        2. List setup requirements
                        3. Present main rules in numbered steps
                        4. Include special rules or exceptions"""
        elif collection_name == "platform_docs":
            system_prompt = """You are a platform support specialist. Provide clear guidance:
                        1. Explain the specific feature or process
                        2. List the steps to accomplish the task
                        3. Include any relevant tips or warnings"""
        else:  # FAQs
            system_prompt = """You are a helpful support assistant. Provide a clear and direct answer:
                        1. Address the specific issue
                        2. Provide the solution
                        3. Include any preventive measures"""

        # Construct the user prompt with context
        user_prompt = f"""Using this context:
                {context_text}

                Answer this question: {query_text}"""

        # Construct API request
        api_request_json = {
            "model": "llama-13b-chat",  # or your preferred model
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False
        }

        # Call API
        response = llama.run(api_request_json)
        response_text = response.json()['choices'][0]['message']['content']

        # Get sources and format response
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
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