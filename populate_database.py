import argparse
import os
import shutil
import time

from chromadb import Settings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import chromadb

CHROMA_PATH = "chroma"
GAME_RULES_PATH = "data/game_rules"
PLATFORM_DOCS_PATH = "data/platform_docs"
FAQ_PATH = "data/faqs"


def safe_clear_database():
    try:
        if os.path.exists(CHROMA_PATH):
            # Force close any existing connections
            try:
                client = chromadb.PersistentClient(
                    path=CHROMA_PATH,
                    settings=Settings(
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                client.reset()
            except Exception as e:
                print(f"Warning: Could not reset client: {e}")

            # Wait a moment
            time.sleep(1)

            # Force delete the directory
            try:
                shutil.rmtree(CHROMA_PATH)
            except Exception as e:
                print(f"Warning: Could not remove directory: {e}")
                # Try to remove files one by one
                for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except:
                            pass
                try:
                    os.rmdir(CHROMA_PATH)
                except:
                    pass

            print("Database cleared successfully")
        else:
            print("No existing database found")

    except Exception as e:
        print(f"Error clearing database: {str(e)}")
        raise


def load_documents_by_type(path: str, doc_type: str):
    print(f"📚 Loading {doc_type} documents from:", path)
    document_loader = PyPDFDirectoryLoader(path)
    documents = document_loader.load()

    print(f"\nLoaded {doc_type} files:")
    sources = set()
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        if source not in sources:
            sources.add(source)
            print(f"- {source}")
    print(f"\nTotal {doc_type} documents loaded: {len(documents)}")

    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], collection_name: str):
    # Initialize Chroma with a specific collection name
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
    )

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add the documents
    existing_items = db.get()
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in {collection_name}: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

    # Process each type of document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    # Load and process game rules
    if os.path.exists(GAME_RULES_PATH):
        game_rules = load_documents_by_type(GAME_RULES_PATH, "game rules")
        game_chunks = text_splitter.split_documents(game_rules)
        add_to_chroma(game_chunks, "game_rules")

    # Load and process platform docs
    if os.path.exists(PLATFORM_DOCS_PATH):
        platform_docs = load_documents_by_type(PLATFORM_DOCS_PATH, "platform docs")
        platform_chunks = text_splitter.split_documents(platform_docs)
        add_to_chroma(platform_chunks, "platform_docs")

    # Load and process FAQs
    if os.path.exists(FAQ_PATH):
        faqs = load_documents_by_type(FAQ_PATH, "FAQs")
        faq_chunks = text_splitter.split_documents(faqs)
        add_to_chroma(faq_chunks, "faqs")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
