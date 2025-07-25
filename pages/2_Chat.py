import streamlit as st
from streamlit_chat import message as st_chat_message
from dotenv import load_dotenv
from agents.supervisor_agent import supervisor_agent
from langsmith import trace, Client

st.set_page_config(page_title="RAG Chatbot", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")

load_dotenv()

client = Client()
supervisor = supervisor_agent()

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize session state for chat feedback
if 'trace_id' not in st.session_state:
    st.session_state.trace_id = ""

with st.expander("**About this page**", expanded=True):
    st.markdown("""
    This is a chatbot that uses a supervisor agent to handle user queries.
    You can ask questions, and the bot will respond based on the provided context.
    The chat history is maintained, and you can provide feedback on the responses.
    Please enter your question in the input box at the bottom of the page.
    """)

with st.expander("**FAQs**", expanded=False):
    st.markdown("""
    **Q: Hi. My name is Mayank**  
      A: Hello Mayank. Nice to meet you.

    **Q: What is 5 + 2?**  
      A: 5 + 2 is 7.

    **Q: What is a large language model?**  
      A: <uses rag_agent to generate the response>.

    **Q: How is the weather of Pune?**  
      A: The weather in Pune is hot and humid.
    """)

# Container for chat history (older messages)
with st.container(height=500):
    idx = None
    for idx, entry in enumerate(st.session_state.history):
        st_chat_message(entry["question"], is_user=True, key=f"user_{idx}", avatar_style=None)
        st_chat_message(entry["answer"], is_user=False, key=f"bot_{idx}", avatar_style=None)

        if entry.get("sources"):
            st.markdown(f"<small><b>Sources:</b> {', '.join(entry['sources'])}</small>", unsafe_allow_html=True)

    if idx is not None:
        selected = st.feedback('stars', key=f'user_feedback_{idx}')
        if selected is not None:
            client.create_feedback(key=f'user_feedback_{idx}', score=selected+1, trace_id=st.session_state.trace_id)
# Spacer to push input to bottom
st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

# Input for user question at the bottom
user_question = st.text_input("Ask a question:", key="input")

result = {"messages": [{}]}
if st.button("Submit"):
    if user_question:
        with st.spinner("Generating response"):
            # Invoke the supervisor agent to handle the question
            with trace("rag_chatbot", inputs={"query": user_question}) as root_run:
                result = supervisor.invoke({"messages": user_question}, config = {"configurable": {"thread_id": "1"}})
                st.session_state.trace_id = root_run.id
                root_run.outputs = result["messages"]
                
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

