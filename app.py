# app.py
import streamlit as st
from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from qa_chain import get_groq_chain_with_history

st.set_page_config(layout="wide")
st.title("Document Q&A Bot")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])

if uploaded_file and openai_api_key and groq_api_key:
    if uploaded_file.name.endswith(".pdf"):
        text = parse_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".csv"):
        text = parse_csv(uploaded_file)

    with st.spinner("Embedding & Indexing..."):
        vs = prepare_vectorstore(text, openai_api_key)
        qa_chain = get_groq_chain_with_history(vs, groq_api_key)
    st.success("Chatbot is ready!")

    query = st.text_input("Ask something about your document:")
    if query:
        # Format chat history
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])

        # Retrieve docs and build input
        docs = vs.similarity_search(query)
        chain_inputs = {
            "question": query,
            "history": history_text,
            "context": "\n".join(doc.page_content for doc in docs)
        }

        answer = qa_chain.combine_documents_chain.run(chain_inputs)
        st.session_state.chat_history.append((query, answer))
        st.markdown(f"**Answer:** {answer}")

    if st.session_state.chat_history:
        st.subheader("🧠 Chat Memory (This Session)")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
