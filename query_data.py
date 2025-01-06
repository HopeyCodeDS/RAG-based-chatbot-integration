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


def query_rag(query_text: str):
    try:
        # Initialize Chroma with a specific collection name
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_function = get_embedding_function()

        db = Chroma(
            client=client,
            collection_name="game_rules",
            embedding_function=embedding_function,
        )

        # Search the DB
        results = db.similarity_search_with_score(query_text, k=5)

        # Prepare context from search results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt = f"""
                    Answer the question based only on the following context:

                    Context:
                    {context_text}

                    ---

                    Question: {query_text}
                    """

        # Construct API request
        api_request_json = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant explaining game rules clearly and concisely."},
                {"role": "user", "content": prompt},
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
