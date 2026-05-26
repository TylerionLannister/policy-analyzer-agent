from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_embeddings():
    """
    Loads a free local HuggingFace embedding model.
    No API key needed — runs locally.
    
    Returns:
        embedding model
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def create_faiss_index(chunks, embeddings, save_path: str = "faiss_index"):
    """
    Embeds document chunks and saves a FAISS index locally.
    Only needs to be run once per document.
    
    Args:
        chunks: list of document chunks from loader.py
        embeddings: embedding model from load_embeddings()
        save_path: where to save the FAISS index
    
    Returns:
        FAISS vectorstore
    """
    print("Embedding chunks... this may take a moment")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to {save_path}/")
    return vectorstore