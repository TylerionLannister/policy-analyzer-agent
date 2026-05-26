from src.loader import load_pdf, chunk_documents
from src.embedder import load_embeddings, create_faiss_index
from src.retriever import load_faiss_index, search_index
from src.tools import search_policy
from src.agent import create_agent
from langchain_core.messages import HumanMessage

#DP0001-0519

#--------------------------------------------------------------
#create embeddings and vectorstore
#pages = load_pdf("data/DP0001-0519.pdf")
#chunks = chunk_documents(pages)
#embeddings = load_embeddings()
#vectorstore = create_faiss_index(chunks, embeddings)

#--------------------------------------------------------------
#load the embeddings, ask 1 query, print chunks
#embeddings = load_embeddings()
#vectorstore = load_faiss_index(embeddings)

#results = search_index(vectorstore, "What property is covered?")

#for i, chunk in enumerate(results):
#    print(f"\n--- Chunk {i+1} ---")
#    print(chunk.page_content)

#--------------------------------------------------------------
#returning the result of the search_policy tool.
#result = search_policy.invoke("Is physical therapy covered?")
#print(result)

#-------------------------------------------------------------
#testing agent

agent = create_agent()

response = agent.invoke({
    "messages": [HumanMessage(content="What kind of property coverage do I have?")]
})

print(response["messages"][-1].content)