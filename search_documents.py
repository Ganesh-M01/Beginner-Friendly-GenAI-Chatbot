
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

project_dir = Path(__file__).parent
database_path = project_dir / "chroma_db"

# Use the same embedding model used during indexing
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the existing vector store
vector_store = Chroma(
    collection_name="company_documents",
    embedding_function=embeddings,
    persist_directory=str(database_path)
)

# Ask a question
question = input("Ask a question about the company: ")

# Retrieve the three most relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

print("\nRelevant document chunks:")

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content)
