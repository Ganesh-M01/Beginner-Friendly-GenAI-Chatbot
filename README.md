
# Beginner-Friendly GenAI Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, LangChain, Groq, Hugging Face embeddings, and ChromaDB. It answers questions using the company information provided to it.

## Features

- Chat interface built with Streamlit
- Retrieval-Augmented Generation (RAG) for document-based question answering
- Groq API integration for language model responses
- Hugging Face sentence embeddings
- ChromaDB vector database for storing and retrieving document embeddings
- Conversation-aware question answering
- Modern dark/purple user interface

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Integration | LangChain and Groq |
| Language Model | `openai/gpt-oss-120b` |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Programming Language | Python |

## Project Structure

```text
Beginner-Friendly-GenAI-Chatbot/
├── app.py
├── company_info.txt
├── requirements.txt
├── chroma_db/
└── .streamlit/
    └── secrets.toml
```

**Note:** Do not upload `secrets.toml` to GitHub. The `chroma_db/` directory should contain the expected indexed data if the application loads an existing vector database.

## Prerequisites

- Python installed
- Groq API key
- Git and GitHub (for deployment)

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Beginner-Friendly-GenAI-Chatbot
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Your `requirements.txt` should include the packages used by your application. A typical starting list is:

```text
streamlit
langchain-groq
langchain-core
langchain-huggingface
langchain-chroma
sentence-transformers
chromadb
```

### 4. Configure the Groq API Key

Create a `.streamlit/secrets.toml` file in the project directory and add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

Get your API key from the [Groq Console](https://console.groq.com/keys).

Never commit API keys or secret files to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will usually open at:

`http://localhost:8501`

## Deploy on Streamlit Community Cloud

1. Push your project to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select your repository, branch, and `app.py`.
4. Open the app's **Settings → Secrets**.
5. Add your Groq API key:

   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key"
   ```

6. Save the settings and deploy the application.

Before deploying, ensure the required documents and vector database are available to the deployed app. If the app expects `chroma_db/` to exist, provide a valid populated database or implement database creation during startup.

Do not publish confidential company information in a public repository or publicly accessible app without authorization.

### Recommended `.gitignore`

```gitignore
.streamlit/secrets.toml
.env
.venv/
venv/
__pycache__/
*.pyc
```

## How It Works

1. **Load Data:** The application uses the configured company information and document sources.
2. **Generate Embeddings:** The Hugging Face embedding model converts text into numerical vectors.
3. **Retrieve Context:** ChromaDB retrieves relevant information for the user's question.
4. **Generate Answers:** LangChain passes the question and retrieved context to the Groq language model.
5. **Handle Follow-up Questions:** The conversation logic can use previous messages to interpret follow-up questions.

The exact ingestion and retrieval behavior depends on the implementation in `app.py` and any supporting files.

## Troubleshooting

### API Key Error

- Verify that `GROQ_API_KEY` is set correctly.
- For local runs, check `.streamlit/secrets.toml`.
- For Streamlit Cloud, check the app's Secrets settings.

### Empty or Irrelevant Answers

- Verify that `company_info.txt` contains the expected information.
- Check that `chroma_db/` exists and contains the correct indexed documents.
- Confirm that the collection name is `company_documents`.
- Re-run the project's indexing process if the database is missing or outdated.

### Missing Package Error

Install the missing package and add it to `requirements.txt` so the deployment environment installs it too.

### Deployment Error

- Check the Streamlit application logs.
- Verify the Python version and package compatibility.
- Ensure all required files are available in the deployed environment.
- Check for dependencies on local-only file paths.

## Security and Privacy

- Store API keys using Streamlit Secrets or environment variables.
- Never commit credentials to source control.
- Do not expose confidential company documents publicly.
- Verify important answers against the original source documents.

## Future Improvements

- Add document upload and indexing functionality.
- Display source citations with generated answers.
- Improve handling of questions with no relevant context.
- Add automated tests for retrieval and answer quality.
- Improve error handling and logging.

## License

Add the appropriate license before distributing or publishing this project.
