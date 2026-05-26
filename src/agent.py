from langchain_groq import ChatGroq


from langchain_core.runnables import RunnablePassthrough

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.tools import search_policy
import os
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    """
    Loads the Groq LLM.
    
    Returns:
        ChatGroq LLM instance
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm


def load_prompt():
    """
    Defines the system prompt and conversation structure for the agent.
    
    Returns:
        ChatPromptTemplate
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful insurance policy analyst assistant. 
        Your job is to help users understand what is and isn't covered in their policy.
        Always use the search_policy tool to find relevant information before answering.
        Be clear, concise, and cite specific policy language when possible.
        If you cannot find relevant information in the policy, say so clearly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return prompt

def create_agent():
    """
    Wires together the LLM, tools, and prompt into an executable agent.
    
    Returns:
        compiled langgraph agent
    """
    llm = load_llm()
    tools = [search_policy]

    agent = create_react_agent(
        llm,
        tools,
        prompt="You are a helpful insurance policy analyst. Always use the search_policy tool before answering. Be clear and cite specific policy language when possible."
    )
    
    return agent