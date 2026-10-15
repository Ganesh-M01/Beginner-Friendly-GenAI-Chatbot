
import os
from getpass import getpass

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = getpass("Enter your Groq API key: ")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Explain concepts clearly."),
    ("human", "{question}")
])

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.3
)

question = input("Ask a question: ")
messages = prompt.format_messages(question=question)
response = model.invoke(messages)

print("\nAssistant:", response.content)
