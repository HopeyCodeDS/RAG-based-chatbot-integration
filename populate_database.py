import argparse
import os
import shutil
import time
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import chromadb

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
