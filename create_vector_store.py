
from pathlib import Path
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. Locate and load the company document
project_dir = Path(__file__).parent
document_path = project_dir / "company_info.txt"
database_path = project_dir / "chroma_db"

if not document_path.exists():
    raise FileNotFoundError(f"Document not found: {document_path}")

text = document_path.read_text(encoding="utf-8")

if not text.strip():
    raise ValueError("company_info.txt is empty.")

print("Document loaded successfully.")
print("Total characters:", len(text))

# 2. Split into larger, overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    add_start_index=True,
)

chunks = splitter.split_text(text)

documents = [
    Document(
        page_content=chunk,
        metadata={
            "source": "company_info.txt",
            "chunk_id": index,
        },
    )
    for index, chunk in enumerate(chunks)
]

print(f"Created {len(documents)} chunks.")

# 3. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# 4. Remove the old database to avoid stale or duplicate chunks
if database_path.exists():
    shutil.rmtree(database_path)
    print("Removed the previous vector database.")

# 5. Create a fresh vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=str(database_path),
    collection_name="company_documents",
)

print("Vector store created successfully!")
print(f"Saved at: {database_path}")
print("Indexed chunks:", len(documents))
