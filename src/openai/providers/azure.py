from __future__ import annotations

import os
import inspect
from typing import Callable, Awaitable
from dataclasses import field, dataclass

import httpx2

from .._models import FinalRequestOptions
from .._provider import _Provider, _create_provider, _ProviderRuntime
from .._exceptions import OpenAIError

AzureTokenProvider = Callable[[], str | Awaitable[str]]


def _validate_credential(value: object) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 33 or ord(char) > 126 for char in value):
        raise OpenAIError("Azure credentials must be non-empty strings suitable for an HTTP header.")
    return value


@dataclass(frozen=True)
class _AzureDefinition:
    base_url: httpx2.URL
    api_key: str | None = field(repr=False)
    azure_ad_token: str | None = field(repr=False)
    token_provider: AzureTokenProvider | None = field(repr=False)
    name: str = field(default="azure", init=False)

    def configure(self) -> _ProviderRuntime:
        def transform(options: FinalRequestOptions) -> FinalRequestOptions:
            # api-key is not stripped by every transport on cross-origin redirects.
            options.follow_redirects = False
            return options

        def validate_request(request: httpx2.Request) -> None:
            url = request.url.copy_with(scheme="https") if request.url.scheme == "wss" else request.url
            origin = (url.scheme, url.host, url.port)
            configured = (self.base_url.scheme, self.base_url.host, self.base_url.port)
            if origin != configured:
                raise OpenAIError("Refusing to send Azure credentials outside the configured origin.")

        def set_credential(request: httpx2.Request, token: object) -> None:
            credential = _validate_credential(token)
            if self.api_key is not None:
                request.headers["api-key"] = credential
            else:
                request.headers["Authorization"] = f"Bearer {credential}"

        def prepare_sync(request: httpx2.Request) -> None:
            validate_request(request)
            token: object = self.api_key if self.api_key is not None else self.azure_ad_token
            if self.token_provider is not None:
                try:
                    token = self.token_provider()
                except Exception:
                    raise OpenAIError("Azure token provider failed.") from None
                if inspect.isawaitable(token):
                    if inspect.iscoroutine(token):
                        token.close()
                    raise OpenAIError("Use AsyncOpenAI with an async Azure token provider.")
            set_credential(request, token)

        async def prepare_async(request: httpx2.Request) -> None:
            validate_request(request)
            token: object = self.api_key if self.api_key is not None else self.azure_ad_token
            if self.token_provider is not None:
                try:
                    token = self.token_provider()
                    if inspect.isawaitable(token):
                        token = await token
                except Exception:
                    raise OpenAIError("Azure token provider failed.") from None
            set_credential(request, token)

        return _ProviderRuntime(
            name=self.name,
            base_url=self.base_url,
            transform_request=transform,
            prepare_request=prepare_sync,
            prepare_async_request=prepare_async,
            prepare_websocket_request=prepare_sync,
            prepare_async_websocket_request=prepare_async,
        )


def azure(
    *,
    endpoint: str | httpx2.URL | None = None,
    api_key: str | None = None,
    azure_ad_token: str | None = None,
    token_provider: AzureTokenProvider | None = None,
) -> _Provider:
    """Configure Azure OpenAI's /openai/v1 HTTP and WebSocket APIs, without a dated api-version.

    endpoint defaults to AZURE_OPENAI_ENDPOINT. Pass exactly one authentication
    option; otherwise AZURE_OPENAI_AD_TOKEN takes precedence over AZURE_OPENAI_API_KEY.
    Prefer azure-identity's get_bearer_token_provider for refreshable Entra tokens.
    Callbacks run on every request attempt; async callbacks require AsyncOpenAI.
    On AsyncOpenAI, synchronous callbacks must not block the event loop.
    """
    endpoint = endpoint if endpoint is not None else os.environ.get("AZURE_OPENAI_ENDPOINT")
    if endpoint is None:
        raise OpenAIError("Azure requires endpoint or AZURE_OPENAI_ENDPOINT.")
    url = httpx2.URL(endpoint)
    if (
        url.scheme != "https"
        or not url.host
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") not in ("", "/openai/v1")
    ):
        raise OpenAIError(
            "Azure endpoint must be an HTTPS resource URL, optionally ending in /openai/v1, "
            "without credentials, query or fragment."
        )
    url = url.copy_with(path="/openai/v1")

    if sum(value is not None for value in (api_key, azure_ad_token, token_provider)) > 1:
        raise OpenAIError("Azure api_key, azure_ad_token and token_provider are mutually exclusive.")
    if api_key is None and azure_ad_token is None and token_provider is None:
        azure_ad_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")
        if azure_ad_token is None:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key is not None:
        _validate_credential(api_key)
    elif azure_ad_token is not None:
        _validate_credential(azure_ad_token)
    elif token_provider is None:
        raise OpenAIError(
            "Azure requires api_key, azure_ad_token or token_provider, "
            "or AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN."
        )
    if token_provider is not None and not callable(token_provider):
        raise OpenAIError("Azure token_provider must be callable.")

    return _create_provider(
        _AzureDefinition(base_url=url, api_key=api_key, azure_ad_token=azure_ad_token, token_provider=token_provider)
    )


__all__ = ["azure", "AzureTokenProvider"]
