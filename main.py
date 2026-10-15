print("=" * 45)
print("     COMPANY KNOWLEDGE ASSISTANT")
print("=" * 45)

user_name = input("Enter your name: ")
question = input("What would you like to know?\n")
category = input("What's the category is this question belongs to?\n Available Categories: Projects, Company Policies, Technical Documentation\n")

print("\n--- Request Summary ---")
print(f"User: {user_name}")
print(f"Question: {question}")
print(f"Category: {category}")

print("\nNext step: Connect an LLM to answer the question.")
