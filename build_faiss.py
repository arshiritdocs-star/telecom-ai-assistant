print("CONFIRMED: USING HUGGINGFACE EMBEDDINGS")

import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- PATHS ----------------
DATA_DIR = "data"                      # folder containing PDFs and text files
DB_DIR = "faiss_db"                    # folder to save FAISS DB

# ---------------- HELPER FUNCTIONS ----------------
def clean_text(text):
    """Remove tables, headings, repeated codes, and messy characters."""
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"G\.\d{3}-G\.\d{3}", "", text)       # ITU table refs
    text = re.sub(r"SYSTEMS ON [A-Z ]+", "", text)      # headings
    text = re.sub(r"[^\w\s,.()-]", "", text)           # other symbols
    return text.strip()

def deduplicate_sentences(text):
    """Remove repeated sentences across the entire dataset."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen = set()
    deduped = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean.lower() not in seen:
            deduped.append(s_clean)
            seen.add(s_clean.lower())
    return " ".join(deduped)

# ---------------- READ ALL DATA ----------------
all_text = ""

print("Reading PDFs and text files from data folder...")

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)
    if file.endswith(".pdf"):
        print(f"Reading PDF: {file}")
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += clean_text(text) + " "
    elif file.endswith(".txt"):
        print(f"Reading text file: {file}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            all_text += clean_text(text) + " "

# ---------------- DEDUPLICATE ----------------
print("Deduplicating sentences across dataset...")
all_text = deduplicate_sentences(all_text)

# ---------------- SPLIT INTO CHUNKS ----------------
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(all_text)
chunks = [c for c in chunks if len(c.split()) > 20]  # discard very short chunks
print(f"Total chunks created: {len(chunks)}")

# ---------------- LOAD HUGGINGFACE EMBEDDINGS ----------------
print("Creating HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- BUILD FAISS INDEX ----------------
print("Building FAISS index...")
db = FAISS.from_texts(chunks, embeddings)

# ---------------- SAVE TO DISK ----------------
db.save_local(DB_DIR)
print("✅ Vector database created successfully!")
print("📁 Saved to folder:", DB_DIR)
