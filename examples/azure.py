from openai import OpenAI
from openai.providers import azure

# Reads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (or AZURE_OPENAI_AD_TOKEN).
client = OpenAI(provider=azure())

completion = client.chat.completions.create(
    model="your-deployment-name",
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)
print(completion.to_json())
