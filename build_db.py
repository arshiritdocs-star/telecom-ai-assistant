print("CONFIRMED: USING HUGGINGFACE EMBEDDINGS")

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
DB_DIR = "faiss_db"

all_text = ""

print("Reading all PDFs from data folder...")

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        print(f"Reading {file}")
        reader = PdfReader(os.path.join(DATA_DIR, file))
        for page in reader.pages:
            if page.extract_text():
                all_text += page.extract_text()

print("Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(all_text)

print("Creating HuggingFace embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS index...")

db = FAISS.from_texts(chunks, embeddings)

db.save_local(DB_DIR)

print("✅ Vector database created successfully!")








