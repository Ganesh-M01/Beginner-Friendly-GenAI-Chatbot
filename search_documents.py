
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

project_dir = Path(__file__).parent

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="company_documents",
    embedding_function=embeddings,
    persist_directory=str(project_dir / "chroma_db"),
)

queries = [
    "Summarize the available company information",
    "What is Syzy Technologies and what services does it provide?",
    "What company projects are listed?",
]

for query in queries:
    print(f"\n{'=' * 60}")
    print("QUERY:", query)

    results = vector_store.similarity_search(query, k=5)

    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(document.page_content)
