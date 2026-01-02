print("✅ USING HUGGINGFACE EMBEDDINGS")

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 📂 Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_DIR = os.path.join(PROJECT_DIR, "faiss_db")

print(f"📂 Looking for PDFs in: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ data folder not found at: {DATA_DIR}")

# 🔍 Collect PDFs
pdf_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(".pdf")
]

print(f"📄 PDFs found: {len(pdf_files)}")

if len(pdf_files) == 0:
    raise ValueError("❌ No PDF files found inside /data folder!")

# 📥 Load pages
docs = []
for path in pdf_files:
    print(f"📥 Loading: {path}")
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

print(f"📑 Total pages loaded: {len(docs)}")

if len(docs) == 0:
    raise ValueError("❌ PDFs contain no extractable text (maybe scanned images).")

# ✂️ Split text
print("✂️ Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)

print(f"🧩 Total chunks created: {len(chunks)}")

if len(chunks) == 0:
    raise ValueError("❌ No chunks created — check PDFs.")

# 🧠 Embeddings
print("🧠 Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 📦 Build DB
print("📦 Building FAISS database...")

db = FAISS.from_documents(chunks, embeddings)

db.save_local(DB_DIR)

print("\n🎉 SUCCESS!")
print(f"📁 Vector DB saved to: {DB_DIR}")


