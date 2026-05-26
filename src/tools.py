from langchain.tools import tool
from src.embedder import load_embeddings
from src.retriever import load_faiss_index, search_index

# Load once at module level so it's not reloaded on every tool call
embeddings = load_embeddings()
vectorstore = load_faiss_index(embeddings)

@tool
def search_policy(query: str) -> str:
    """
    Searches the insurance policy document for information relevant to the query.
    Use this when the user asks whether something is covered, excluded, or defined
    in the policy. Input should be a natural language question.
    """
    results = search_index(vectorstore, query, k=5)
    
    if not results:
        return "No relevant information found in the policy document."
    
    # Format chunks into a single string for the LLM
    formatted = "\n\n".join([
        f"[Chunk {i+1}]:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return formatted