#from langchain_pypdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Loads a PDF and returns raw pages.
    
    Args:
        pdf_path: path to the PDF file
    
    Returns:
        list of raw pages
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages


def chunk_documents(pages, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits raw pages into smaller chunks for embedding.
    
    Args:
        pages: list of documents from load_pdf
        chunk_size: number of characters per chunk
        chunk_overlap: overlap between chunks to preserve context
    
    Returns:
        list of document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks