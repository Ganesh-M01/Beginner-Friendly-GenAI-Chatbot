
import os
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Company AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS - BLACK & PURPLE THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #09090F;
        color: #F4F0FF;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                ellipse at 50% 0%,
                rgba(124, 58, 237, 0.14),
                transparent 50%
            ),
            #09090F;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 9, 15, 0.75);
    }

    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    h1, h2, h3 {
        color: #F5F3FF !important;
        letter-spacing: -0.5px;
    }

    p {
        color: #C4C0D3;
    }

    [data-testid="stCaptionContainer"] {
        color: #A6A0B8;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background: #0D0B14;
        border-right: 1px solid #292238;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #0D0B14;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #EDE9FE !important;
    }

    [data-testid="stSidebar"] p {
        color: #A9A3BC;
    }

    /* Buttons */

    .stButton > button {
        background: linear-gradient(
            135deg,
            #7C3AED,
            #6D28D9
        );
        color: #FFFFFF !important;
        border: 1px solid #8B5CF6;
        border-radius: 11px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #8B5CF6,
            #7C3AED
        );
        border-color: #C4B5FD;
        box-shadow: 0 0 18px rgba(139, 92, 246, 0.2);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.35);
    }

    /* Chat messages */

    [data-testid="stChatMessage"] {
        background: rgba(19, 17, 29, 0.9);
        border: 1px solid #292238;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.9rem;
        transition: border-color 0.2s ease;
    }

    [data-testid="stChatMessage"]:hover {
        border-color: #493568;
    }

    [data-testid="stChatMessage"] p {
        color: #E5E1F0;
        line-height: 1.75;
    }

    /* Chat input */

    [data-testid="stChatInput"] {
        border: 1px solid #3A2A55 !important;
        border-radius: 16px !important;
        background: #12101B !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 1px #8B5CF6;
    }

    [data-testid="stChatInput"] textarea {
        color: #F5F3FF !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777187 !important;
    }

    .stTextInput input,
    .stTextArea textarea {
        background: #12101B !important;
        color: #F5F3FF !important;
        border: 1px solid #342A4A !important;
        border-radius: 10px !important;
    }

    /* Dividers and alerts */

    hr {
        border-color: #292238 !important;
    }

    [data-testid="stAlert"] {
        background: #171020;
        border: 1px solid #49326C;
        border-radius: 12px;
    }

    a {
        color: #A78BFA !important;
    }

    /* Scrollbar */

    ::-webkit-scrollbar {
        width: 7px;
    }

    ::-webkit-scrollbar-track {
        background: #09090F;
    }

    ::-webkit-scrollbar-thumb {
        background: #45335F;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #8B5CF6;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 6rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

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


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

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
        persist_directory=str(database_path),
    )

    return vector_store


@st.cache_resource
def load_model(api_key):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
    )


try:
    with st.spinner("Initializing your AI assistant..."):
        vector_store = load_rag_components()
        model = load_model(api_key)

except Exception as error:
    st.error(f"Unable to initialize the AI assistant: {error}")
    st.stop()


# ============================================================
# PROMPTS AND CHAINS
# ============================================================

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the latest user question as a standalone question
for searching company documents.

Use conversation history to resolve references such as
"it", "they", "these", and "that".
Preserve the original meaning.
Return only the rewritten question.""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

rewrite_chain = rewrite_prompt | model | StrOutputParser()



answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and conversational company knowledge assistant.

Your task is to answer the user's actual question using the
provided document context and conversation history.

Rules:
1. Always address the user's actual request.
2. If the user asks to summarize, summarize the relevant information
   available in the context directly.
3. If the context contains general company information, provide
   a useful overview instead of claiming that information is missing.
4. If only some information is available, summarize what is available
   and clearly mention any specific gaps.
5. Never invent facts or claim that information is unavailable
   when the context actually contains it.
6. Use clear, natural language and avoid unnecessary disclaimers.
7. For company-specific facts, rely on the retrieved context.

Retrieved document context:
{context}""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

answer_chain = answer_prompt | model | StrOutputParser()


# ============================================================
# SIDEBAR
# Use st.html() to render HTML directly.
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="padding:12px 0 20px;">
            <div style="
                display:flex;
                align-items:center;
                gap:12px;
            ">
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    width:44px;
                    height:44px;
                    flex-shrink:0;
                    border-radius:13px;
                    background:linear-gradient(
                        135deg,#7C3AED,#4C1D95
                    );
                    color:#FFFFFF;
                    font-size:23px;
                ">
                    ✦
                </div>

                <div>
                    <div style="
                        color:#F5F3FF;
                        font-size:16px;
                        font-weight:700;
                    ">
                        SYZY AI
                    </div>

                    <div style="
                        color:#9690A8;
                        font-size:11px;
                        margin-top:3px;
                        letter-spacing:0.7px;
                    ">
                        COMPANY KNOWLEDGE
                    </div>
                </div>
            </div>
        </div>
        """
    )

    st.divider()

    st.markdown("### 💬 Conversation")

    st.caption(
        f"{len(st.session_state.messages) // 2} "
        "completed exchanges in this session"
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.markdown("### 📚 Knowledge Base")

    st.html(
        """
        <div style="
            background:#15111F;
            border:1px solid #342746;
            border-radius:12px;
            padding:13px;
        ">
            <div style="
                color:#C4B5FD;
                font-size:13px;
                font-weight:600;
            ">
                ● Syzy Technologies
            </div>

            <div style="
                color:#9690A8;
                font-size:12px;
                margin-top:7px;
                line-height:1.6;
            ">
                Answers are generated using relevant information
                from your company documents.
            </div>
        </div>
        """
    )

    st.divider()

    st.markdown("### ⚙️ Technology")
    st.caption("• Streamlit — Interface")
    st.caption("• LangChain — RAG pipeline")
    st.caption("• Chroma — Vector database")
    st.caption("• Hugging Face — Embeddings")
    st.caption("• Groq — Language model")

    st.html(
        """
        <div style="
            margin-top:30px;
            padding-top:15px;
            border-top:1px solid #292238;
            color:#777187;
            font-size:11px;
            line-height:1.7;
        ">
            Powered by AI<br>
            Company Knowledge Assistant
        </div>
        """
    )


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div style="text-align:center; padding:1.2rem 0 2rem;">

        <div style="
            display:inline-block;
            padding:7px 15px;
            margin-bottom:18px;
            background:rgba(124,58,237,0.10);
            border:1px solid #49326C;
            border-radius:30px;
            color:#C4B5FD;
            font-size:11px;
            font-weight:600;
            letter-spacing:1.5px;
        ">
            ✦ SYZY TECHNOLOGIES
        </div>

        <h1 style="
            font-size:clamp(32px,5vw,46px);
            font-weight:750;
            line-height:1.2;
            margin:0;
            background:linear-gradient(
                90deg,#E9D5FF,#A78BFA,#7C3AED
            );
            -webkit-background-clip:text;
            background-clip:text;
            -webkit-text-fill-color:transparent;
        ">
            Company AI Assistant
        </h1>

        <p style="
            color:#A6A0B8;
            font-size:15px;
            margin-top:14px;
            line-height:1.7;
        ">
            Your intelligent companion for company knowledge.<br>
            Ask questions and explore your documents.
        </p>

    </div>
    """
)


# ============================================================
# WELCOME SCREEN AND SUGGESTED QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.html(
        """
        <div style="
            background:linear-gradient(
                145deg,
                rgba(124,58,237,0.09),
                rgba(18,16,27,0.85)
            );
            border:1px solid #302442;
            border-radius:20px;
            padding:27px 24px;
            margin:0 auto 25px;
            text-align:center;
        ">
            <div style="font-size:34px; margin-bottom:10px;">
                ✨
            </div>

            <div style="
                color:#F5F3FF;
                font-size:20px;
                font-weight:650;
                margin-bottom:9px;
            ">
                What would you like to know?
            </div>

            <div style="
                color:#A6A0B8;
                font-size:14px;
                line-height:1.8;
            ">
                Ask about Syzy Technologies, its projects, technologies,
                or information in your indexed company documents.
            </div>
        </div>
        """
    )

    st.markdown(
        "<p style='font-size:13px; color:#A6A0B8; "
        "margin-bottom:12px;'>TRY ASKING</p>",
        unsafe_allow_html=True,
    )

    suggestions = [
        "What projects are in the knowledge base?",
        "What technologies are mentioned in the documents?",
        "Summarize the available company information.",
        "What details can you find about a project?",
    ]

    suggestion_columns = st.columns(2)

    for index, suggestion in enumerate(suggestions):

        with suggestion_columns[index % 2]:

            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion
                st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Message your company AI assistant..."
)

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:

    # Display and save the user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer.
    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Searching company documents and generating your answer..."
            ):

                # Rewrite follow-up questions.
                standalone_question = rewrite_chain.invoke(
                    {
                        "chat_history": st.session_state.chat_history[-10:],
                        "question": question,
                    }
                )

                # Retrieve relevant company documents.
                documents = vector_store.similarity_search(
                    standalone_question,
                    k=3,
                )

                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )

                # Generate the response.
                answer = answer_chain.invoke(
                    {
                        "context": context,
                        "chat_history": st.session_state.chat_history[-10:],
                        "question": question,
                    }
                )

            st.markdown(answer)

            # Save messages for future turns.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            st.session_state.chat_history.extend(
                [
                    HumanMessage(content=question),
                    AIMessage(content=answer),
                ]
            )

        except Exception as error:
            st.error(
                f"Something went wrong while generating the answer: {error}"
            )
