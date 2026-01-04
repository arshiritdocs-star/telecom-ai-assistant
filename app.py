import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# App Settings
# -----------------------------
DB_DIR = "faiss_db"

st.set_page_config(page_title="📡 Telecom Knowledge Chatbot", layout="wide")
st.title("📡 Telecom Knowledge Chatbot (Offline & API-Free)")

# -----------------------------
# Check FAISS DB
# -----------------------------
if not os.path.exists(DB_DIR):
    st.error("❌ FAISS DB not found. Run build_faiss.py first.")
    st.stop()

# -----------------------------
# Load Embeddings + FAISS DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

st.success("✅ Vector database loaded")

# -----------------------------
# Safety-Focused Prompt
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable telecom assistant.

Use ONLY the information in the context to answer.
If the answer is not found in the context, reply:

"I do not have enough information from the documents to answer that."

Be clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# Local HuggingFace Model (No API)
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",   # upgraded from small → better accuracy
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=generator)

# -----------------------------
# Retrieval QA Chain
# -----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}   # retrieve more docs for accuracy
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# Streamlit UI
# -----------------------------
query = st.text_input("🔍 Ask a telecom question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = qa.run(query)

            st.markdown("### 🟢 Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
