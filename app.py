import os
import streamlit as st
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
   # updated
from langchain.chains import RetrievalQA
              # updated
from langchain_community.llms import HuggingFacePipeline        # updated
from transformers import pipeline




DB_DIR = "faiss_db"

st.set_page_config(page_title="📡 Telecom Knowledge Chatbot", layout="wide")
st.title("📡 Telecom Knowledge Chatbot (No API Keys)")

# -----------------------------
# Check FAISS DB
# -----------------------------
if not os.path.exists(DB_DIR):
    st.error("❌ FAISS DB not found. Run build_faiss.py first.")
    st.stop()

st.success("✅ FAISS DB Loaded")

# -----------------------------
# Load embeddings & FAISS
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Prompt Template
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a telecom expert.
Answer simply and clearly using the context below.
Do NOT copy text directly.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# Local LLM (No API Key)
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=generator)

# -----------------------------
# Retrieval QA chain
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
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
            st.error(f"Error: {e}")


