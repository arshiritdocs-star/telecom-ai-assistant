import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_DIR = os.path.join(PROJECT_DIR, "faiss_db")

def build_faiss_if_missing():
    """
    Builds FAISS vectorstore from PDFs in data/ if faiss_db/ does not exist.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ data folder not found at: {DATA_DIR}")

    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if len(pdf_files) == 0:
        raise ValueError("❌ No PDF files found inside /data folder!")

    # Load PDFs
    docs = []
    for path in pdf_files:
        print(f"📥 Loading: {path}")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    if len(docs) == 0:
        raise ValueError("❌ PDFs contain no extractable text (maybe scanned images).")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)

    if len(chunks) == 0:
        raise ValueError("❌ No chunks created — check PDFs.")

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS DB
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_DIR)

    print("🎉 SUCCESS! FAISS database created at:", DB_DIR)
