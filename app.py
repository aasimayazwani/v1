import streamlit as st
from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from qa_chain import get_groq_chain

st.set_page_config(layout="wide")
st.title("📄🔍 Groq-Powered Document Q&A Bot")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

uploaded_file = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])

if uploaded_file and openai_api_key and groq_api_key:
    if uploaded_file.name.endswith(".pdf"):
        text = parse_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".csv"):
        text = parse_csv(uploaded_file)

    with st.spinner("Embedding & Indexing..."):
        vs = prepare_vectorstore(text, openai_api_key)
        qa_chain = get_groq_chain(vs, groq_api_key)
    st.success("Chatbot is ready!")

    query = st.text_input("Ask something about your document:")
    if query:
        answer = qa_chain.run(query)
        st.markdown(f"**Answer:** {answer}")