
# Reusable prompt template

template = """
You are a helpful AI assistant.

Task: Summarize the project in {word_count} words.
Audience: {audience}

Project details:
{project_details}
"""

# Dynamic values
prompt = template.format(
    word_count=100,
    audience="a beginner",
    project_details="An AI application that searches company documents "
                    "and answers questions using relevant information."
)

print(prompt)
