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

# ---------------- PATHS ----------------
DB_DIR = "faiss_db"

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
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- LOAD FAISS ----------------
@st.cache_resource
def load_db():
    embeddings = load_embeddings()
    if not os.path.exists(DB_DIR):
        st.warning("FAISS database not found. Only glossary definitions are available.")
        return None
    return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

embeddings = load_embeddings()
db = load_db()

# ---------------- CLEAN & REFINE ----------------
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"G\.\d{3}-G\.\d{3}", "", text)
    text = re.sub(r"SYSTEMS ON [A-Z ]+", "", text)
    text = re.sub(r"[^\w\s,.()-]", "", text)
    return text.strip()

def expand_abbreviations(text):
    abbreviations = {"POTS": "Plain Old Telephone Service (POTS)",
                     "GPON": "Gigabit Passive Optical Network (GPON)",
                     "XGS-PON": "10-Gigabit Symmetric Passive Optical Network (XGS-PON)"}
    for abbr, full in abbreviations.items():
        text = re.sub(rf"\b{abbr}\b", full, text)
    return text

def refine_text(text, query, top_n=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s for s in sentences if len(s.split()) > 5]  # discard very short sentences
    keywords = query.lower().split()
    ranked = sorted(sentences, key=lambda s: sum(k in s.lower() for k in keywords), reverse=True)
    cleaned = [clean_text(s) for s in ranked[:top_n]]
    return expand_abbreviations("\n\n".join(cleaned))

# ---------------- LOAD FREE LLM ----------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"  # Slightly bigger CPU-friendly model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return generator

text_generator = load_llm()

def generate_answer(prompt):
    output = text_generator(prompt, max_length=500, do_sample=True, temperature=0.7)
    return output[0]['generated_text']

# ---------------- USER INPUT ----------------
query = st.text_input("Enter your telecom question:")

if query:
    key = query.lower().strip()
    # ---------------- GLOSSARY FALLBACK ----------------
    if key in GLOSSARY:
        st.subheader("📡 Glossary Definition")
        st.write(GLOSSARY[key])
    elif db:
        # ---------------- FAISS RETRIEVAL ----------------
        results = db.similarity_search_with_score(query, k=5)
        
        # Filter and refine top chunks
        top_docs_raw = [d.page_content for d, _ in sorted(results, key=lambda x: x[1], reverse=True)]
        top_docs = []
        for doc in top_docs_raw:
            refined = refine_text(doc, query, top_n=3)
            if len(refined.split()) > 30:  # discard very short chunks
                top_docs.append(refined)
        
        # Merge top 3 chunks for LLM context
        context = "\n\n".join(top_docs[:3])

        if not context.strip():
            context = "No complete information found. Showing best-match excerpts."

        # ---------------- STRUCTURED PROMPT ----------------
        prompt = f"""
You are a telecom expert. Based on the context below, write a detailed, human-readable paragraph answering the question.
Include:
- Definition
- Key Points / Types
- Examples (if relevant)
Do NOT copy sentences verbatim; synthesize into a coherent explanation.

Context:
{context}

Question:
{query}
"""
        answer = generate_answer(prompt)
        st.subheader("📡 Telecom Expert Answer")
        st.write(answer)

        # ---------------- SOURCE PASSAGES ----------------
        st.subheader("📎 Source Passages")
        for i, doc in enumerate(top_docs, start=1):
            with st.expander(f"Source {i}"):
                st.write(doc)
    else:
        st.warning("No FAISS database found. Only glossary definitions are available.")
