# chatbot.py
# Offline Telecom Knowledge Assistant (RAG-based)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "faiss_db"

print("📡 Loading telecom knowledge base...")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS DB
db = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Telecom AI Assistant is ready!")
print("Type 'exit' to quit\n")

while True:
    query = input("Ask a telecom question: ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    docs = db.similarity_search(query, k=3)

    print("\n📘 Relevant Information from Knowledge Base:\n")

    for i, doc in enumerate(docs, start=1):
        print(f"--- Source {i} ---")
        print(doc.page_content)
        print()

    print("-" * 60)


