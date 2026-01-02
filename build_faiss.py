import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
from pdf2image import convert_from_path
import pytesseract

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_DIR = os.path.join(PROJECT_DIR, "faiss_db")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ data folder not found at: {DATA_DIR}")

# -----------------------------
# Helper: extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = " ".join([doc.page_content for doc in docs]).strip()

    if text:
        return docs
    else:
        # OCR fallback
        print(f"⚡ OCR fallback for scanned PDF: {pdf_path}")
        pages = convert_from_path(pdf_path)
        ocr_docs = []
        for page in pages:
            page_text = pytesseract.image_to_string(page)
            if page_text.strip():
                ocr_docs.append(Document(page_content=page_text))
        return ocr_docs

# -----------------------------
# Load PDFs + OCR
# -----------------------------
pdf_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    raise ValueError("❌ No PDFs found in data folder!")

docs = []
for pdf_path in pdf_files:
    print(f"📥 Processing {pdf_path}")
    docs.extend(extract_text_from_pdf(pdf_path))

# Optional: add extra manual prompts
extra_texts = [
    "Telecommunication industry overview, trends, and future opportunities."
]
docs.extend([Document(page_content=t) for t in extra_texts])

# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"🧩 Total chunks created: {len(chunks)}")

# -----------------------------
# Embeddings + FAISS
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
db.save_local(DB_DIR)
print(f"🎉 FAISS DB saved at: {DB_DIR}")
