import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

# -----------------------------
# Paths
# -----------------------------
DB_DIR = "faiss_db"
st.title("📡 Telecom Knowledge Chatbot")

# -----------------------------
# File uploader for PDFs/images
# -----------------------------
uploaded_files = st.file_uploader(
    "📂 Upload PDFs or images to add to context (optional)", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

def ocr_file(file):
    """Run OCR on PDF or image files."""
    text = ""
    if file.name.lower().endswith(".pdf"):
        images = convert_from_bytes(file.read())
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
    else:
        img = Image.open(file)
        text += pytesseract.image_to_string(img)
    return text

uploaded_text = ""
if uploaded_files:
    for f in uploaded_files:
        uploaded_text += ocr_file(f) + "\n"
    st.success("✅ Uploaded files processed with OCR")

# -----------------------------
# Check FAISS DB
# -----------------------------
if not os.path.exists(DB_DIR):
    st.warning("❌ FAISS DB not found. You can still use uploaded files for QA.")
    vectorstore = None
else:
    st.success("✅ Found FAISS DB")
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

# -----------------------------
# Streamlit QA function
# -----------------------------
query = st.text_input("🔍 Ask your telecom question:")

if query:
    with st.spinner("Thinking..."):
        try:
            context = uploaded_text
            if vectorstore:
                # Combine FAISS retriever context with uploaded OCR text
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                faiss_context = " ".join([doc.page_content for doc in retriever.get_relevant_documents(query)])
                context = faiss_context + "\n" + uploaded_text

            answer = llm(prompt=prompt.format(context=context, question=query))
            st.markdown(f"### 🟢 Answer\n{answer}")
        except Exception as e:
            st.error(f"Error: {e}")
