
from pathlib import Path
import getpass

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Project paths
project_dir = Path(__file__).parent
database_path = project_dir / "chroma_db"

# 2. Groq API key
api_key = getpass.getpass("Enter your Groq API key: ").strip()

# 3. Initialize the LLM
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.2
)

# 4. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Load the existing vector store
vector_store = Chroma(
    collection_name="company_documents",
    embedding_function=embeddings,
    persist_directory=str(database_path)
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. Rewrite follow-up questions for document retrieval
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the latest user question as a standalone question
that can be searched in a company knowledge base.

Use the conversation history to resolve references such as
"it", "they", "these", and "what about that".
Preserve the user's original intent.
If the question is already standalone, keep its meaning unchanged.
Return only the rewritten question."""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

rewrite_chain = rewrite_prompt | model | StrOutputParser()

# 7. Prompt for the final answer
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a company knowledge assistant.

Use the retrieved document context to answer questions about
the company. Use conversation history to understand follow-up
questions and maintain continuity.

Do not invent company-specific facts.
If the answer is not available in the retrieved context,
say that the provided documents do not contain that information.

Retrieved context:
{context}"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

answer_chain = answer_prompt | model | StrOutputParser()

# 8. In-memory conversational history
chat_history = []

print("\nCompany RAG Assistant with Memory is ready!")
print("Type 'exit' to quit.")
print("Type 'clear' to clear conversation history.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if question.lower() == "clear":
        chat_history.clear()
        print("Conversation history cleared.\n")
        continue

    if not question:
        continue

    try:
        # A. Resolve follow-up references using conversation history
        standalone_question = rewrite_chain.invoke({
            "chat_history": chat_history[-10:],
            "question": question
        })

        # B. Retrieve relevant document chunks
        documents = retriever.invoke(standalone_question)

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        # C. Generate an answer using context and history
        answer = answer_chain.invoke({
            "context": context,
            "chat_history": chat_history[-10:],
            "question": question
        })

        print("\nAssistant:", answer, "\n")

        # D. Save this turn for future questions
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

    except Exception as error:
        print(f"\nAn error occurred: {error}\n")
