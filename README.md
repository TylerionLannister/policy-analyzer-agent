# Insurance Policy Analyzer

An AI agent that answers questions about insurance policy documents using RAG and LangChain.

## Stack
- LangChain + LangGraph
- Groq (Llama 3.1)
- FAISS
- HuggingFace Embeddings
- Streamlit

## Setup
1. Clone the repo
2. Create a virtual environment and install dependencies
   pip install -r requirements.txt
3. Add a .env file with your GROQ_API_KEY
4. Add a policy PDF to /data
5. Run the embedder once to create the FAISS index
6. Launch the app
   streamlit run app.py
