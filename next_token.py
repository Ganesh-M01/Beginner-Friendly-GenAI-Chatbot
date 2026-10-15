
# A simplified demonstration of next-token prediction

next_word = {
    "The capital of France is": "Paris",
    "Python is a programming": "language",
    "RAG helps an LLM retrieve": "information",
    "AI agents can call": "tools"
}

prompt = input("Enter a prompt: ").strip()

if prompt in next_word:
    print("Predicted next token:", next_word[prompt])
else:
    print("No prediction available for this prompt.")
    print("Try one of the example prompts.")
