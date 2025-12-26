# app.py
# Offline Telecom AI Web Application using Streamlit + FAISS

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

DB_DIR = "faiss_db"  # folder containing your saved FAISS DB

# ---------------- UI SETUP ----------------
st.set_page_config(
    page_title="Telecom AI Assistant",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Telecom AI Assistant")
st.write("Ask questions related to Telecommunication, GPON, XGS-PON, and broadband systems.")

# ---------------- LOAD EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

# ---------------- LOAD DATABASE ----------------
@st.cache_resource
def load_db():
    embeddings = load_embeddings()

    if not os.path.exists(DB_DIR):
        st.error("FAISS database not found. Please make sure 'faiss_db' folder exists.")
        return None

    db = FAISS.load_local(
        DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


embeddings = load_embeddings()
db = load_db()

# ---------------- GLOSSARY ----------------
GLOSSARY = {
    "gpon": "Gigabit Passive Optical Network — a fiber broadband technology delivering high-speed data using passive splitters.",
    "xgs-pon": "XGS-PON — 10-Gigabit symmetric Passive Optical Network offering 10 Gbps downlink and uplink.",
    "olt": "Optical Line Terminal — the service-provider side device controlling PON access.",
    "onu": "Optical Network Unit — the subscriber-side fiber termination device.",
    "ont": "Optical Network Terminal — customer-premises fiber modem.",
    "bandwidth": "Maximum data transfer rate of a network connection.",
    "latency": "Delay between sending and receiving data."
}

# ----------- Extract relevant sentences ----------
def extract_relevant_sentences(text, query):
    sentences = re.split(r'(?<=[.!?]) +', text)
    keywords = query.lower().split()

    ranked = sorted(
        sentences,
        key=lambda s: sum(k in s.lower() for k in keywords),
        reverse=True
    )

    return " ".join(ranked[:3])

# ---------------- USER INPUT ----------------
query = st.text_input("Enter your telecom question:")

# ---------------- PROCESS QUERY ----------------
if query and db:

    query_emb = embeddings.embed_query(query)
    results = db.similarity_search_with_score(query, k=8)

    rescored = []
    for doc, _ in results:
        doc_emb = embeddings.embed_query(doc.page_content)
        sim = cosine_similarity([query_emb], [doc_emb])[0][0]
        rescored.append((doc, sim))

    rescored = sorted(rescored, key=lambda x: x[1], reverse=True)

    if len(rescored) == 0:
        st.error("No matching telecom reference data found in your database.")
    else:
        top_score = rescored[0][1]

        if top_score < 0.10:
            st.warning("Low-confidence match. Try rephrasing your question.")
        
        docs = [d for d,_ in rescored[:3]]

        def extract_relevant_sentences(text, query):
            sentences = re.split(r'(?<=[.!?]) +', text)
            if len(sentences) == 1:
                return text
            keywords = query.lower().split()
            ranked = sorted(
                sentences,
                key=lambda s: sum(k in s.lower() for k in keywords),
                reverse=True
            )
            return " ".join(ranked[:4])

        context = "\n\n".join(
            extract_relevant_sentences(d.page_content, query)
            for d in docs
        )

        if not context.strip():
            context = "No exact definition found. Displaying best-match telecom reference excerpts."

        st.subheader("📡 Telecom Expert Answer")
        st.write(context)

        st.subheader("📎 Source Passages")
        for i, doc in enumerate(docs, start=1):
            with st.expander(f"Source {i}"):
                st.write(doc.page_content)
