import streamlit as st
from streamlit_chat import message as st_chat_message
from langchain import hub
from dotenv import load_dotenv
from agents.supervisor_agent import supervisor_agent

st.set_page_config(page_title="RAG Chatbot", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")

load_dotenv()

supervisor = supervisor_agent()

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Container for chat history (older messages)
with st.container(height=500):
    for idx, entry in enumerate(st.session_state.history):
        st_chat_message(entry["question"], is_user=True, key=f"user_{idx}", avatar_style=None)
        st_chat_message(entry["answer"], is_user=False, key=f"bot_{idx}", avatar_style=None)
        if entry.get("sources"):
            st.markdown(f"<small><b>Sources:</b> {', '.join(entry['sources'])}</small>", unsafe_allow_html=True)

# Spacer to push input to bottom
st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

# Input for user question at the bottom
user_question = st.text_input("Ask a question:", key="input")

if st.button("Submit"):
    if user_question:
        with st.spinner("Generating response"):
            # Invoke the supervisor agent to handle the question
            result = supervisor.invoke({"messages": user_question}, config = {"configurable": {"thread_id": "1"}})

            # Add a dummy div for scrolling
            st.markdown(
                "<div id='scroll-anchor'></div>",
                unsafe_allow_html=True
            )
        # Store the question and answer in history
        st.session_state.history.append({
            "question": user_question,
            "answer": result["messages"][-1].content,
            "sources": result.get("sources", [])
        })
        st.rerun()
    else:
        st.warning("Please enter a question.")