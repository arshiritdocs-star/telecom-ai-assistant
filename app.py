import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

DB_DIR = "faiss_db"

st.set_page_config(page_title="📡 Telecom Knowledge Chatbot", layout="wide")
st.title("📡 Telecom Knowledge Chatbot (No API Keys)")

# -----------------------------------
# Check FAISS DB exists
# -----------------------------------
if not os.path.exists(DB_DIR):
    st.error("❌ FAISS DB not found. Run build_faiss.py first.")
    st.stop()
st.success("✅ FAISS DB Loaded")

# -----------------------------------
# Load Embeddings + Vector Store
# -----------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------------
# Strong Anti-Hallucination Prompt
# -----------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a telecom expert assistant.

Think briefly before answering.

Use ONLY the information provided in the context below.
If the answer is not found in the context, reply:
"I do not have enough information to answer that."

Provide a short, complete explanation (1–2 sentences) rather than a single word.
Format the answer as:
- Bold the term being defined
- Give a 1–2 sentence definition
- Add a short example if possible

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------------
# Local LLM — no API
# -----------------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # base model for better quality
    max_new_tokens=128,
    min_length=40,                # ensures at least 1–2 sentences
    temperature=0.1,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=generator)

# -----------------------------------
# Improved MMR Retrieval
# -----------------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "fetch_k": 40},  # more candidate chunks for better context
    search_type="mmr"
)

# -----------------------------------
# RetrievalQA Chain
# -----------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

# -----------------------------------
# Streamlit UI
# -----------------------------------
query = st.text_input("🔍 Ask a telecom question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = qa.run(query)
            st.markdown("### 🟢 Answer")
            st.markdown(answer)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
