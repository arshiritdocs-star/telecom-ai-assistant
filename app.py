import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


DB_DIR = "faiss_db"


# --------------------------------------------------
# Streamlit UI setup
# --------------------------------------------------
st.set_page_config(page_title="📡 Telecom Knowledge Chatbot", layout="wide")
st.title("📡 Telecom Knowledge Chatbot (No API Keys)")


# --------------------------------------------------
# Verify FAISS DB Exists
# --------------------------------------------------
if not os.path.exists(DB_DIR):
    st.error("❌ FAISS DB not found. Run build_faiss.py first.")
    st.stop()
else:
    st.success("✅ FAISS database loaded successfully")


# --------------------------------------------------
# Load embeddings & database
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)


# --------------------------------------------------
# Strong Anti-Hallucination Prompt
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a telecom expert assistant.

Think briefly before answering.

Use ONLY the information in the context below.
If the answer is not found in the context, say:
"I do not have enough information to answer that."

Keep the explanation clear, simple, and factual.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# Local LLM — Google FLAN-T5-BASE
# --------------------------------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=128,
    temperature=0.1,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=generator)


# --------------------------------------------------
# Improved Retriever (MMR)
# --------------------------------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    },
    search_type="mmr"
)


# --------------------------------------------------
# Retrieval QA Chain
# --------------------------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)


# --------------------------------------------------
# Streamlit Chat UI
# --------------------------------------------------
query = st.text_input("🔍 Ask a telecom question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = qa.run(query)

            st.markdown("### 🟢 Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
