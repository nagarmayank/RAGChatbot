import streamlit as st
from streamlit_chat import message as st_chat_message
from rag_app import RAGAgent
from langchain.chat_models import init_chat_model
from langchain import hub

st.set_page_config(page_title="RAG Chatbot", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")
model = init_chat_model("gemini-2.5-flash-preview-05-20", model_provider="google_genai", temperature=0.7)
prompt = hub.pull("rlm/rag-prompt")

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Container for chat history (older messages)
with st.container(height=500):
    for idx, entry in enumerate(st.session_state.history):
        st_chat_message(entry["question"], is_user=True, key=f"user_{idx}", avatar_style=None)
        st_chat_message(entry["answer"], is_user=False, key=f"bot_{idx}", avatar_style=None)

# Spacer to push input to bottom
st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

# Input for user question at the bottom
user_question = st.text_input("Ask a question:", key="input")

if st.button("Submit"):
    if user_question:
        with st.spinner("Generating response"):
            # Invoke the RAG agent
            rag_agent = RAGAgent(model, prompt)
            result = rag_agent.graph.invoke({"question": user_question})

            # Add a dummy div for scrolling
            st.markdown(
                "<div id='scroll-anchor'></div>",
                unsafe_allow_html=True
            )
        # Store the question and answer in history
        st.session_state.history.append({
            "question": user_question,
            "answer": result["answer"]
        })
        st.rerun()
    else:
        st.warning("Please enter a question.")