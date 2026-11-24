import asyncio

from openai import OpenAI, AsyncOpenAI
from openai.providers import azure

scopes = "https://cognitiveservices.azure.com/.default"

deployment_name = "your-deployment-name"


def sync_main() -> None:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    with DefaultAzureCredential() as credential:
        token_provider = get_bearer_token_provider(credential, scopes)
        # Reads AZURE_OPENAI_ENDPOINT for endpoint selection
        client = OpenAI(provider=azure(token_provider=token_provider))
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": "How do I output all files in a directory using Python?",
                }
            ],
        )
        print(completion.to_json())


async def async_main() -> None:
    from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

    async with DefaultAzureCredential() as credential:
        token_provider = get_bearer_token_provider(credential, scopes)
        # Reads AZURE_OPENAI_ENDPOINT for endpoint selection
        client = AsyncOpenAI(provider=azure(token_provider=token_provider))
        completion = await client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": "How do I output all files in a directory using Python?",
                }
            ],
        )
        print(completion.to_json())


sync_main()

asyncio.run(async_main())
