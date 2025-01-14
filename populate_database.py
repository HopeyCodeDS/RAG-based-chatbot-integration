import os
import traceback
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def safe_remove_dir_contents(directory: str):
    """Safely remove directory contents without removing the directory itself"""
    logger.info(f"Attempting to clear contents of {directory}")
    try:
        if os.path.exists(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                        logger.info(f"Removed file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        logger.info(f"Removed directory: {item_path}")
                except Exception as e:
                    logger.warning(f"Could not remove {item_path}: {e}")
    except Exception as e:
        logger.warning(f"Error clearing directory {directory}: {e}")


def load_documents():
    """Load documents from the data directory"""
    logger.info(f"📚 Loading documents from: {DATA_PATH}")
    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = document_loader.load()

        logger.info("\nLoaded files:")
        sources = set()
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.add(source)
                logger.info(f"- {source}")
        logger.info(f"\nTotal documents loaded: {len(documents)}")

        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise


def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise


def add_to_chroma(documents: List) -> None:
    """Add documents to Chroma collections based on their source paths"""
    try:
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
            game_db = Chroma(
                client=client,
                collection_name="game_rules",
                embedding_function=embedding_function,
            )
            game_db.add_documents(game_docs)
            logger.info(f"Added {len(game_docs)} documents to game_rules collection")

        # Handle platform documents
        if platform_docs:
            platform_db = Chroma(
                client=client,
                collection_name="platform_docs",
                embedding_function=embedding_function,
            )
            platform_db.add_documents(platform_docs)
            logger.info(f"Added {len(platform_docs)} documents to platform_docs collection")

    except Exception as e:
        logger.error(f"Error adding documents to Chroma: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main function to populate the database"""
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--reset", action="store_true", help="Reset the database")
        args = parser.parse_args()

        if args.reset:
            logger.info("✨ Clearing Database")
            safe_remove_dir_contents(CHROMA_PATH)

        # Load and process documents
        documents = load_documents()
        chunks = split_documents(documents)

        # Get existing document count
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            game_collection = client.get_collection("game_rules")
            platform_collection = client.get_collection("platform_docs")
            existing_count = len(game_collection.get()['ids']) + len(platform_collection.get()['ids'])
        except Exception:
            existing_count = 0

        logger.info(f"Number of existing documents in DB: {existing_count}")
        logger.info(f"👉 Adding new documents: {len(chunks)}")

        # Add to database
        add_to_chroma(chunks)
        logger.info("✅ Database population completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()