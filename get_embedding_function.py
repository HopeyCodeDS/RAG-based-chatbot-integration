from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    # Using a lightweight model that's good for text embeddings
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings