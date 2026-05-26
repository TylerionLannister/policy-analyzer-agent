from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_faiss_index(embeddings, index_path: str = "faiss_index"):
    """
    Loads a previously saved FAISS index from disk.
    
    Args:
        embeddings: same embedding model used to create the index
        index_path: path to the saved FAISS index
    
    Returns:
        FAISS vectorstore
    """
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"FAISS index loaded from {index_path}/")
    return vectorstore


def search_index(vectorstore, query: str, k: int = 3):
    """
    Searches the FAISS index for chunks relevant to the query.
    
    Args:
        vectorstore: loaded FAISS vectorstore
        query: natural language question
        k: number of chunks to retrieve
    
    Returns:
        list of relevant document chunks
    """
    results = vectorstore.similarity_search(query, k=k)
    return results