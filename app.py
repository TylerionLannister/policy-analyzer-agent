import streamlit as st
from src.agent import create_agent
from langchain_core.messages import HumanMessage

# Page config
st.set_page_config(
    page_title="Policy Analyzer",
    page_icon="📋",
    layout="centered"
)

st.title("📋 Insurance Policy Analyzer")
st.markdown("Upload an insurance policy and ask questions about your coverage.")

# Initialize agent once and store in session state
if "agent" not in st.session_state:
    with st.spinner("Loading agent..."):
        st.session_state.agent = create_agent()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your policy..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching policy..."):
            response = st.session_state.agent.invoke({
                "messages": [HumanMessage(content=prompt)]
            })
            answer = response["messages"][-1].content
            st.markdown(answer)

    # Add response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })