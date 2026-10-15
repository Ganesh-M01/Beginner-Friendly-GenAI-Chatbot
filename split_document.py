
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

# Step 3: Split the document
chunks = splitter.split_text(text)

# Step 4: Display the results
print("\nNumber of chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
