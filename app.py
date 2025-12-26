import unzip_data  # runs only once

import os
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="Telecom AI Assistant", page_icon="📡", layout="centered")
st.title("📡 Telecom AI Assistant")
st.write("Ask questions related to Telecommunication, GPON, XGS-PON, and broadband systems.")

DB_DIR = "faiss_db"

# ---------------- GLOSSARY ----------------
GLOSSARY = {
    "gpon": "Gigabit Passive Optical Network — a fiber broadband system delivering high-speed internet using passive optical splitters.",
    "xgs-pon": "10-Gigabit symmetric Passive Optical Network providing 10 Gbps downstream and upstream.",
    "olt": "Optical Line Terminal — the central office device controlling PON traffic.",
    "onu": "Optical Network Unit — subscriber-side fiber access device.",
    "ont": "Optical Network Terminal — customer-premises fiber modem.",
    "bandwidth": "Maximum rate at which data can be transferred.",
    "latency": "Delay between sending and receiving data."
}

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_db():
    embeddings = load_embeddings()
    if not os.path.exists(DB_DIR):
        st.warning("FAISS database not found.")
        return None
    return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

embeddings = load_embeddings()
db = load_db()
generator = load_llm()

# ---------------- CLEANING ----------------
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)

    # remove technical headings & code artifacts
    text = re.sub(r"G\.\d{3}.*?\s", "", text)
    text = re.sub(r"SYSTEMS ON [A-Z \-]+", "", text)
    text = re.sub(r"[^\w\s,.()\-]", "", text)

    return text.strip()

def rank_human_readable(chunks):
    """Prefer readable sources like Wikipedia / Britannica / reports."""
    priority_words = [
        "telecommunications", "is a", "refers to", "service providers",
        "pon", "xgs", "india", "market", "customer", "technology", "broadband"
    ]

    def score(text):
        t = text.lower()
        return sum(word in t for word in priority_words)

    return sorted(chunks, key=score, reverse=True)

def generate_answer(context, question):
    prompt = f"""
You are a friendly telecom expert. Explain clearly using simple language.
Avoid repeating technical ITU wording. Write fresh sentences.

Context:
{context}

Question:
{question}
"""
    output = generator(prompt, max_length=450, temperature=0.6)
    return output[0]["generated_text"]

# ---------------- UI INPUT ----------------
query = st.text_input("Enter your telecom question:")

if query:
    key = query.lower().strip()

    # glossary first
    if key in GLOSSARY:
        st.subheader("📘 Glossary Answer")
        st.write(GLOSSARY[key])

    elif db:
        results = db.similarity_search_with_score(query, k=8)

        # lowest score = closest match
        results = sorted(results, key=lambda x: x[1])

        raw_chunks = [clean_text(doc.page_content) for doc, _ in results]

        readable_chunks = rank_human_readable(raw_chunks)

        # pick best 3
        context = "\n\n".join(readable_chunks[:3])

        answer = generate_answer(context, query)

        st.subheader("📡 Telecom Expert Answer")
        st.write(answer)

        st.subheader("📎 Key Source Passages")
        for i, doc in enumerate(readable_chunks[:3], start=1):
            with st.expander(f"Source {i}"):
                st.write(doc)

    else:
        st.warning("Database missing — only glossary works.")

