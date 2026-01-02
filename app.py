import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from build_faiss import build_faiss_if_missing

DB_DIR = "faiss_db"

st.set_page_config(
    page_title="📡 Telecom Knowledge Chatbot",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Knowledge Chatbot")

# -----------------------------
# 1. Make sure FAISS DB exists
# -----------------------------
if not os.path.exists(DB_DIR):
    st.warning("FAISS database not found. Building it from PDFs...")
    with st.spinner("⏳ Building FAISS database... This may take a few minutes."):
        build_faiss_if_missing()
    st.success("✅ FAISS database created!")

else:
    st.success("✅ FAISS database found.")

# -----------------------------
# 2. Load embeddings + DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# 3. Prepare prompt template
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a telecom expert.
Use the context to answer clearly and simply.
Do NOT copy the text word-for-word.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# 4. Load LLM
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
# 5. Streamlit UI
# -----------------------------
query = st.text_input("🔍 Ask your telecom question:")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            answer = qa.run(query)
            st.markdown(f"### 🟢 Answer\n{answer}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
