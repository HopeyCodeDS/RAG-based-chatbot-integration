import argparse
import os
from typing import List
import shutil
import time

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def load_documents():
    print("📚 Loading documents from:", DATA_PATH)
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()

    print("\nLoaded files:")
    sources = set()
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        if source not in sources:
            sources.add(source)
            print(f"- {source}")
    print(f"\nTotal documents loaded: {len(documents)}")

    return documents


def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(documents: List):
    # # Initialize Chroma with a specific collection name
    # client = chromadb.PersistentClient(path=CHROMA_PATH)
    #
    # db = Chroma(
    #     client=client,
    #     collection_name="game_rules",
    #     embedding_function=get_embedding_function(),
    # )
    #
    # # Calculate Page IDs
    # chunks_with_ids = calculate_chunk_ids(chunks)
    #
    # # Add the documents
    # existing_items = db.get()
    # existing_ids = set(existing_items["ids"])
    # print(f"Number of existing documents in DB: {len(existing_ids)}")
    #
    # # Only add documents that don't exist in the DB
    # new_chunks = []
    # for chunk in chunks_with_ids:
    #     if chunk.metadata["id"] not in existing_ids:
    #         new_chunks.append(chunk)
    #
    # if len(new_chunks):
    #     print(f"👉 Adding new documents: {len(new_chunks)}")
    #     new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    #     db.add_documents(new_chunks, ids=new_chunk_ids)
    # else:
    #     print("✅ No new documents to add")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = get_embedding_function()

    # Separate documents by type
    game_docs = []
    platform_docs = []

    for doc in documents:
        source_path = doc.metadata['source']
        if 'game_rules' in source_path:
            game_docs.append(doc)
        elif 'platform_docs' in source_path:
            platform_docs.append(doc)

    logger.info(f"Found {len(game_docs)} game documents and {len(platform_docs)} platform documents")

    # Handle game rules documents
    if game_docs:
        # Use Langchain's Chroma wrapper
        game_db = Chroma(
            client=client,
            collection_name="game_rules",
            embedding_function=embedding_function,
        )
        game_db.add_documents(game_docs)
        logger.info(f"Added {len(game_docs)} documents to game_rules collection")

    # Handle platform documents
    if platform_docs:
        # Use Langchain's Chroma wrapper
        platform_db = Chroma(
            client=client,
            collection_name="platform_docs",
            embedding_function=embedding_function,
        )
        platform_db.add_documents(platform_docs)
        logger.info(f"Added {len(platform_docs)} documents to platform_docs collection")

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

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
