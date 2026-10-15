
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the document
project_dir = Path(__file__).parent
document_path = project_dir / "company_info.txt"

text = document_path.read_text(encoding="utf-8")

# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Created {len(chunks)} text chunks.")

# 3. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# 4. Create and save the vector store
database_path = project_dir / "chroma_db"

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory=str(database_path),
    collection_name="company_documents"
)

print("Vector store created successfully!")
print(f"Saved at: {database_path}")
