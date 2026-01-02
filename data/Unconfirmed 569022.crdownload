# app.py
# Offline Telecom AI Web Application using Streamlit + FAISS

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DB_DIR = "faiss_db"  # folder relative to app.py

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

# --- Robust FAISS DB loader ---
@st.cache_resource
def load_db():
    embeddings = load_embeddings()

    # Debug info
    st.write("DB folder exists:", os.path.exists(DB_DIR))
    if os.path.exists(DB_DIR):
        st.write("Files in DB folder:", os.listdir(DB_DIR))

    try:
        db = FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.write("✅ FAISS DB loaded successfully!")
    except Exception as e:
        st.write("⚠️ FAISS DB failed to load. Rebuilding...")
        st.write("Error:", e)

        # Rebuild FAISS index from documents
        # TODO: Replace with your actual document loading logic
        documents = []  # placeholder
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(DB_DIR)
        st.write("✅ FAISS DB rebuilt and saved.")

    return db
# --- End of loader ---

embeddings = load_embeddings()
db = load_db()

query = st.text_input("Enter your telecom question:")

def clean(text):
    bad_tokens = ["................................................................",
                  ".....", "Code Point", "Figure", "Table", "Index",
                  "ISBN", "copyright"]
    for t in bad_tokens:
        text = text.replace(t, " ")
    return " ".join(text.split())

if query:

    # ---- smarter retrieval ----
    results = db.similarity_search_with_score(query, k=8)

    filtered_docs = []
    for doc, score in results:
        if score < 0.45:  # keep only strong matches
            filtered_docs.append(doc)

    docs = filtered_docs[:3]  # limit to top 3

    # ---- merge & clean ----
    context = "\n\n".join([clean(d.page_content) for d in docs])

    # ---- create refined answer ----
    final_answer = f"""
Telecom Expert Answer:

Based on trusted telecom standards and technical references,
here is a clear explanation related to your question:

{context}

Summary:
The retrieved information discusses topics that directly relate 
to your query. Technical noise and formatting were removed for clarity.
"""

    st.subheader("📘 Telecom Answer")
    st.write(final_answer)

    # ---- show sources for transparency ----
    st.subheader("📎 Sources Used")
    for i, doc in enumerate(docs, start=1):
        with st.expander(f"Source {i}"):
            st.write(doc.page_content)


