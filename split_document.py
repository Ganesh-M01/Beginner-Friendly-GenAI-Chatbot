
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Locate and read the document
file_path = Path(__file__).parent / "company_info.txt"
text = file_path.read_text(encoding="utf-8")

print("Document loaded successfully!")
print("Total characters:", len(text))

# Step 2: Configure the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# Step 3: Split the do
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
cument
chunks = splitter.split_text(text)

# Step 4: Display the results
print("\nNumber of chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
