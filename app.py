
import os
import streamlit as st

from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Company AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Company AI Assistant")
st.caption("Ask questions about Syzy Technologies")



# ---------- API KEY ----------
api_key = os.getenv("GROQ_API_KEY", "")

try:
    api_key = api_key or st.secrets.get("GROQ_API_KEY", "")
except (FileNotFoundError, RuntimeError):
    pass

if not api_key:
    
    st.error(
        "Groq API key not found. Set GROQ_API_KEY "
        "in PowerShell or configure Streamlit secrets."
    )
    st.stop()

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- LOAD RAG COMPONENTS ----------
@st.cache_resource
def load_rag_components():
    project_dir = Path(__file__).parent
    database_path = project_dir / "chroma_db"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="company_documents",
        embedding_function=embeddings,
        persist_directory=str(database_path)
    )

    return vector_store


@st.cache_resource
def load_model(api_key):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2
    )


vector_store = load_rag_components()
model = load_model(api_key)


# ---------- PROMPTS ----------
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the latest user question as a standalone question
for searching company documents.

Use conversation history to resolve references such as
"it", "they", "these", and "that".
Preserve the original meaning.
Return only the rewritten question."""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

rewrite_chain = rewrite_prompt | model | StrOutputParser()


answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a conversational company knowledge assistant.

Use conversation history to understand follow-up questions
and recall previous messages.

Use retrieved document context for company-specific facts.
If the requested company information is not in the context,
say that it could not be found in the provided documents.
Do not invent company facts.

Retrieved context:
{context}"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

answer_chain = answer_prompt | model | StrOutputParser()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Chat Settings")
    st.write("Knowledge base: Syzy Technologies")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Powered by Streamlit, LangChain, Chroma and Groq.")


# ---------- DISPLAY CHAT HISTORY ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- CHAT INPUT ----------
question = st.chat_input("Ask a question about the company...")

if question:
    # Display and store the user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):

                # Rewrite follow-up questions
                standalone_question = rewrite_chain.invoke({
                    "chat_history": st.session_state.chat_history[-10:],
                    "question": question
                })

                # Retrieve relevant company documents
                documents = vector_store.similarity_search(
                    standalone_question,
                    k=3
                )

                context = "\n\n".join(
                    doc.page_content for doc in documents
                )

                # Generate the response
                answer = answer_chain.invoke({
                    "context": context,
                    "chat_history": st.session_state.chat_history[-10:],
                    "question": question
                })

            st.markdown(answer)

            # Save messages for future turns
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state.chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])

        except Exception as error:
            st.error(f"Something went wrong: {error}")
