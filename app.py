import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# -----------------------------
# Paths
# -----------------------------
DB_DIR = "faiss_db"
st.title("📡 Telecom Knowledge Chatbot")

# -----------------------------
# Check FAISS DB
# -----------------------------
if not os.path.exists(DB_DIR):
    st.error("❌ FAISS DB not found. Run build_faiss.py first.")
    st.stop()

st.success("✅ Found FAISS DB")

# -----------------------------
# Load embeddings + FAISS
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

# -----------------------------
# Prompt template
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a telecom expert.
Use the context to answer clearly and simply.
Do NOT copy text word-for-word.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# Local LLM (no API keys)
# -----------------------------
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# Streamlit UI
# -----------------------------
query = st.text_input("🔍 Ask your telecom question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = qa.run(query)
            st.markdown(f"### 🟢 Answer\n{answer}")
        except Exception as e:
            st.error(f"Error: {e}")
