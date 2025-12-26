# app.py
# Free Telecom AI Assistant using Streamlit + FAISS + HuggingFace Embeddings

import os
import re
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "faiss_db"  # folder containing your saved FAISS DB

# ---------------- UI SETUP ----------------
st.set_page_config(
    page_title="Telecom AI Assistant",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Telecom AI Assistant")
st.write("Ask questions related to Telecommunication, GPON, XGS-PON, and broadband systems.")

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

# ---------------- LOAD EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------- LOAD DATABASE ----------------
@st.cache_resource
def load_db():
    embeddings = load_embeddings()
    if not os.path.exists(DB_DIR):
        st.warning("FAISS database not found. Only glossary definitions will be available.")
        return None
    db = FAISS.load_local(
        DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

embeddings = load_embeddings()
db = load_db()

# ---------------- TEXT CLEANING & REFINEMENT ----------------
def clean_text(text):
    """Clean raw technical text."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"G\.\d{3}-G\.\d{3}", "", text)           # remove ITU-T style headers
    text = re.sub(r"SYSTEMS ON [A-Z ]+", "", text)          # remove all caps headings
    text = re.sub(r"[^\w\s,.()-]", "", text)               # remove special characters
    return text.strip()

def expand_abbreviations(text):
    """Expand key telecom abbreviations."""
    abbreviations = {
        "POTS": "Plain Old Telephone Service (POTS)",
        "GPON": "Gigabit Passive Optical Network (GPON)",
        "XGS-PON": "10-Gigabit Symmetric Passive Optical Network (XGS-PON)"
    }
    for abbr, full in abbreviations.items():
        text = re.sub(rf"\b{abbr}\b", full, text)
    return text

def refine_text(text, query, top_n=5):
    """
    Extract relevant sentences and make them human-readable.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s for s in sentences if len(s.split()) > 5]  # remove short fragments

    keywords = query.lower().split()
    ranked = sorted(
        sentences,
        key=lambda s: sum(k in s.lower() for k in keywords),
        reverse=True
    )

    cleaned = [clean_text(s) for s in ranked[:top_n]]
    answer = "\n\n".join(cleaned)
    answer = expand_abbreviations(answer)
    return answer

# ---------------- USER INPUT ----------------
query = st.text_input("Enter your telecom question:")

# ---------------- PROCESS QUERY ----------------
if query:
    # Check glossary first
    key = query.lower().strip()
    if key in GLOSSARY:
        st.subheader("📡 Glossary Definition")
        st.write(GLOSSARY[key])
    elif db:
        # Query FAISS database
        query_emb = embeddings.embed_query(query)
        results = db.similarity_search_with_score(query, k=8)

        rescored = []
        for doc, _ in results:
            doc_emb = embeddings.embed_query(doc.page_content)
            sim = cosine_similarity([query_emb], [doc_emb])[0][0]
            rescored.append((doc, sim))

        rescored = sorted(rescored, key=lambda x: x[1], reverse=True)
        docs = [d for d,_ in rescored[:3]]

        if not docs:
            st.error("No matching reference data found in your database.")
        else:
            # Combine and refine top documents
            context = " ".join(
                refine_text(d.page_content, query, top_n=3)
                for d in docs
            )

            if not context.strip():
                context = "No exact definition found. Showing best-match excerpts."

            st.subheader("📡 Telecom Expert Answer")
            st.write(context)

            st.subheader("📎 Source Passages")
            for i, doc in enumerate(docs, start=1):
                with st.expander(f"Source {i}"):
                    st.write(clean_text(doc.page_content))
    else:
        st.warning("No FAISS database found. Only glossary definitions are available.")
