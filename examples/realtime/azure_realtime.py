import os
import asyncio

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from openai import AsyncOpenAI
from openai.providers import azure

# Azure OpenAI Realtime Docs

# How-to: https://learn.microsoft.com/azure/ai-services/openai/how-to/realtime-audio
# Supported models and API versions: https://learn.microsoft.com/azure/ai-services/openai/how-to/realtime-audio#supported-models
# Entra ID auth: https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity


async def main() -> None:
    """The following example demonstrates how to configure Azure OpenAI to use the Realtime API.
    For an audio example, see push_to_talk_app.py and update the client and model parameter accordingly.

    When prompted for user input, type a message and hit enter to send it to the model.
    Enter "q" to quit the conversation.
    """

    async with DefaultAzureCredential() as credential:
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

        # The endpoint of your Azure OpenAI resource is required. You can set it in the AZURE_OPENAI_ENDPOINT
        # environment variable.
        # You can find it in the Microsoft Foundry portal in the Overview page of your Azure OpenAI resource.
        # Example: https://{your-resource}.openai.azure.com
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

        deployment_name = "your-deployment-name"

        async with (
            AsyncOpenAI(provider=azure(endpoint=endpoint, token_provider=token_provider)) as client,
            client.realtime.connect(model=deployment_name) as connection,
        ):
            await connection.session.update(
                session={
                    "output_modalities": ["text"],
                    "model": deployment_name,
                    "type": "realtime",
                },
            )
            while True:
                user_input = input("Enter a message: ")
                if user_input == "q":
                    break

                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_input}],
                    }
                )
                await connection.response.create()
                async for event in connection:
                    if event.type == "response.output_text.delta":
                        print(event.delta, flush=True, end="")
                    elif event.type == "response.output_text.done":
                        print()
                    elif event.type == "error":
                        raise RuntimeError(f"Realtime error: {event.error.type} ({event.error.code})")
                    elif event.type == "response.done":
                        if event.response.status != "completed":
                            raise RuntimeError(f"Response did not complete successfully: {event.response.status}")
                        break
                else:
                    raise RuntimeError("WebSocket closed before the response completed.")


asyncio.run(main())
