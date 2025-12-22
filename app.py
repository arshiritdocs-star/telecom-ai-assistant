# app.py
# Offline Telecom AI Web Application using Streamlit + FAISS

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "faiss_db"

st.set_page_config(
    page_title="Telecom AI Assistant",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Telecom AI Assistant")
st.write("Ask questions related to Telecommunication, GPON, XGS-PON, and broadband systems.")

# Load embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Load FAISS database
@st.cache_resource
def load_db():
    embeddings = load_embeddings()
    return FAISS.load_local(
        DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

embeddings = load_embeddings()
db = load_db()

query = st.text_input("Enter your telecom question:")

if query:
    docs = db.similarity_search(query, k=3)

    st.subheader("📘 Relevant Information from Knowledge Base:")

    for i, doc in enumerate(docs, start=1):
        with st.expander(f"Source {i}"):
            st.write(doc.page_content)

st.markdown("---")
st.caption("Offline AI system using FAISS vector database and telecom standards")
