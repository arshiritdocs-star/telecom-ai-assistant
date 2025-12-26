print("CONFIRMED: USING HUGGINGFACE EMBEDDINGS")

import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- PATHS ----------------
DATA_DIR = "data"                      # folder containing PDFs
WIKI_FILE = "telecom_wiki.txt"         # Wikipedia text
BRITANNICA_FILE = "telecom_britannica.txt"  # Britannica text
DB_DIR = "faiss_db"                    # folder to save FAISS DB

# ---------------- HELPER FUNCTIONS ----------------
def clean_text(text):
    """Clean raw text by removing line breaks, numbering, tables, and unwanted symbols"""
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"G\.\d{3}-G\.\d{3}", "", text)
    text = re.sub(r"SYSTEMS ON [A-Z ]+", "", text)
    text = re.sub(r"[^\w\s,.()-]", "", text)
    return text.strip()

def deduplicate_sentences(text):
    """Remove duplicate sentences"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen = set()
    deduped = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            deduped.append(s_clean)
            seen.add(s_clean)
    return " ".join(deduped)

# ---------------- READ PDF TEXT ----------------
all_text = ""
print("Reading all PDFs from data folder...")
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        print(f"Reading {file} ...")
        reader = PdfReader(os.path.join(DATA_DIR, file))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += clean_text(text) + "\n\n"

# ---------------- READ WIKI TEXT ----------------
if os.path.exists(WIKI_FILE):
    print(f"Reading Wikipedia content from {WIKI_FILE} ...")
    with open(WIKI_FILE, "r", encoding="utf-8") as f:
        wiki_text = f.read()
        all_text += "\n\n" + deduplicate_sentences(clean_text(wiki_text))
else:
    print("No Wikipedia file found. Skipping.")

# ---------------- READ BRITANNICA TEXT ----------------
if os.path.exists(BRITANNICA_FILE):
    print(f"Reading Britannica content from {BRITANNICA_FILE} ...")
    with open(BRITANNICA_FILE, "r", encoding="utf-8") as f:
        brit_text = f.read()
        all_text += "\n\n" + deduplicate_sentences(clean_text(brit_text))
else:
    print("No Britannica file found. Skipping.")

# ---------------- SPLIT INTO CHUNKS ----------------
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(all_text)
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
