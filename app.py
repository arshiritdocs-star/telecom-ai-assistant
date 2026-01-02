import os
import streamlit as st

# -----------------------------
# 1. Build FAISS DB on startup
# -----------------------------
st.info("Checking FAISS database...")

try:
    import build_faiss   # runs automatically if needed
except Exception as e:
    st.error(f"FAISS build failed: {e}")
    st.stop()


# -----------------------------
# 2. Correct imports
# -----------------------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub


DB_DIR = "faiss_db"


# -----------------------------
# 3. Load FAISS DB
# -----------------------------
if not os.path.exists(DB_DIR):
    st.error("❌ FAISS DB not found. Make sure build_faiss.py created it.")
    st.stop()

st.success("✅ FAISS database found.")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------
# 4. Custom Prompt
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a telecom expert.

Use the context to answer clearly and simply.
Do NOT copy text directly.
Explain in human-friendly language.

Context:
{context}

Question:
{question}

Answer:
"""
)


# -----------------------------
# 5. LLM (FREE HuggingFace Hosted)
# -----------------------------
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0}
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)


# -----------------------------
# 6. Streamlit UI
# -----------------------------
st.title("📡 Telecom Knowledge Chatbot")

st.write("Ask any telecom-related question below:")

query = st.text_input("🔍 Your Question:")

if query:
    with st.spinner("Thinking…"):
        try:
            answer = qa.run(query)
            st.markdown(f"### 🟢 Answer\n{answer}")
        except Exception as e:
            st.error(f"Error: {e}")
