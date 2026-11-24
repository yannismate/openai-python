from __future__ import annotations

import inspect
import traceback
from typing import Any
from unittest.mock import Mock, AsyncMock

import httpx2
import pytest

from openai import OpenAI, AsyncOpenAI, OpenAIError, APIStatusError, omit
from openai.providers import azure


@pytest.fixture(autouse=True)
def azure_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-unrelated-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example/v1")


@pytest.mark.parametrize(
    ("asynchronous", "mode"),
    [
        (False, "api_key"),
        (False, "azure_ad_token"),
        (False, "token_provider"),
        (True, "api_key"),
        (True, "azure_ad_token"),
        (True, "token_provider"),
        (True, "async_token_provider"),
    ],
)
async def test_authentication_routing_refresh_and_copy(
    asynchronous: bool, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    requests: list[httpx2.Request] = []
    tokens = iter(["fake-first-token", "fake-retry-token", "fake-copy-token"])

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx2.Response(429, json={}, headers={"retry-after-ms": "1"})
        return httpx2.Response(200, json={"id": "resp_fake", "object": "response", "output": []})

    async def async_token() -> str:
        return next(tokens)

    options: dict[str, Any]
    if mode == "async_token_provider":
        options = {"token_provider": async_token}
    elif mode == "token_provider":
        options = {"token_provider": lambda: next(tokens)}
    else:
        options = {mode: "fake-static-credential"}
    provider = azure(endpoint="https://azure.example/", **options)
    transport = httpx2.MockTransport(handle)
    client = (
        AsyncOpenAI(provider=provider, http_client=httpx2.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(provider=provider, http_client=httpx2.Client(transport=transport))
    )
    try:
        response = client.responses.create(model="deployment-name", input="hello")
        if inspect.isawaitable(response):
            response = await response
        assert response.id == "resp_fake"
        copied = client.with_options(timeout=10)
        response = copied.responses.retrieve("resp_fake")
        if inspect.isawaitable(response):
            await response
    finally:
        closed = client.close()
        if inspect.isawaitable(closed):
            await closed

    assert [str(request.url) for request in requests] == [
        "https://azure.example/openai/v1/responses",
        "https://azure.example/openai/v1/responses",
        "https://azure.example/openai/v1/responses/resp_fake",
    ]
    header = "api-key" if mode == "api_key" else "authorization"
    other_header = "authorization" if mode == "api_key" else "api-key"
    expected = (
        ["fake-first-token", "fake-retry-token", "fake-copy-token"]
        if mode.endswith("token_provider")
        else ["fake-static-credential"] * 3
    )
    assert [request.headers[header] for request in requests] == [
        value if mode == "api_key" else f"Bearer {value}" for value in expected
    ]
    assert all(other_header not in request.headers for request in requests)
    assert all(value not in caplog.text for value in expected)
    assert "fake-static-credential" not in repr(provider)


@pytest.mark.parametrize(
    ("key", "token", "expected"),
    [
        (None, None, None),
        ("fake-key", None, ("api-key", "fake-key")),
        ("fake-key", "fake-token", ("authorization", "Bearer fake-token")),
    ],
)
def test_environment_authentication(
    monkeypatch: pytest.MonkeyPatch, key: str | None, token: str | None, expected: tuple[str, str] | None
) -> None:
    if key is not None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", key)
    if token is not None:
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", token)
    if expected is None:
        with pytest.raises(OpenAIError, match="Azure requires"):
            azure()
        return
    response = httpx2.Response(200, json={"data": []})
    with OpenAI(
        provider=azure(),
        http_client=httpx2.Client(transport=httpx2.MockTransport(lambda _: response)),
    ) as client:
        client.models.list()
    assert response.request.headers[expected[0]] == expected[1]


def test_explicit_credentials_ignore_ambient_azure_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-ambient-key")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "fake-ambient-token")
    response = httpx2.Response(200, json={"data": []})
    with OpenAI(
        provider=azure(api_key="fake-explicit-key"),
        http_client=httpx2.Client(transport=httpx2.MockTransport(lambda _: response)),
    ) as client:
        client.models.list()
    assert response.request.headers["api-key"] == "fake-explicit-key"
    assert "authorization" not in response.request.headers


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": "fake-key", "azure_ad_token": "fake-token"},
        {"api_key": "fake-key", "token_provider": lambda: "fake-token"},
        {"azure_ad_token": "fake-token", "token_provider": lambda: "fake-token"},
        {"api_key": ""},
        {"azure_ad_token": "fake-token\r\ninjected: header"},
        {"token_provider": "not-callable"},
    ],
)
def test_invalid_authentication_options(options: dict[str, Any]) -> None:
    with pytest.raises(OpenAIError):
        azure(**options)


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://azure.example",
        "https://user:fake-secret@azure.example",
        "https://azure.example?key=fake-secret",
        "https://azure.example/#fragment",
        "https://azure.example/openai/deployments/test",
    ],
)
def test_invalid_endpoint(endpoint: str) -> None:
    with pytest.raises(OpenAIError, match="Azure endpoint"):
        azure(endpoint=endpoint, api_key="fake-key")


def test_v1_endpoint_is_not_appended_twice() -> None:
    with OpenAI(provider=azure(endpoint="https://azure.example/openai/v1/", api_key="fake-key")) as client:
        assert str(client.base_url) == "https://azure.example/openai/v1/"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("api_key", [False, True])
async def test_credentials_cannot_escape_origin_or_follow_redirects(asynchronous: bool, api_key: bool) -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(307, headers={"location": "https://other.example/models"})

    provider = azure(api_key="fake-key") if api_key else azure(token_provider=lambda: "fake-token")
    transport = httpx2.MockTransport(handle)
    client = (
        AsyncOpenAI(provider=provider, http_client=httpx2.AsyncClient(transport=transport, follow_redirects=True))
        if asynchronous
        else OpenAI(provider=provider, http_client=httpx2.Client(transport=transport, follow_redirects=True))
    )
    try:
        with pytest.raises(APIStatusError) as error:
            result = client.get("/models", cast_to=httpx2.Response)
            if inspect.isawaitable(result):
                await result
        assert error.value.status_code == 307
        assert len(requests) == 1
        with pytest.raises(OpenAIError, match="outside the configured origin"):
            result = client.get("https://other.example/models", cast_to=httpx2.Response)
            if inspect.isawaitable(result):
                await result
        assert len(requests) == 1
    finally:
        closed = client.close()
        if inspect.isawaitable(closed):
            await closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("header", ["Authorization", "API-KEY"])
async def test_custom_authentication_is_rejected_before_token_resolution(asynchronous: bool, header: str) -> None:
    def unexpected_token() -> str:
        pytest.fail("Conflicting requests must not resolve credentials")

    provider = azure(token_provider=unexpected_token)
    transport = httpx2.MockTransport(lambda _: pytest.fail("Conflicting requests must not be sent"))
    client = (
        AsyncOpenAI(provider=provider, http_client=httpx2.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(provider=provider, http_client=httpx2.Client(transport=transport))
    )
    try:
        with pytest.raises(OpenAIError, match="owns authentication"):
            result = client.models.list(extra_headers={header: "fake-custom"})
            if inspect.isawaitable(result):
                await result
    finally:
        closed = client.close()
        if inspect.isawaitable(closed):
            await closed


def test_sync_client_rejects_async_token_provider() -> None:
    async def token_provider() -> str:
        return "fake-token"

    with OpenAI(provider=azure(token_provider=token_provider)) as client:
        with pytest.raises(OpenAIError, match="Use AsyncOpenAI"):
            client.models.list()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["empty", "wrong-type", "header-injection", "exception"])
async def test_token_failures_are_explicit_and_redacted(asynchronous: bool, failure: str) -> None:
    def token_provider() -> Any:
        if failure == "exception":
            raise RuntimeError("fake-private-credential")
        return {"empty": "", "wrong-type": None, "header-injection": "fake-private-credential\r\n"}[failure]

    provider = azure(token_provider=token_provider)
    transport = httpx2.MockTransport(lambda _: pytest.fail("Invalid credentials must not be sent"))
    client = (
        AsyncOpenAI(provider=provider, http_client=httpx2.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(provider=provider, http_client=httpx2.Client(transport=transport))
    )
    try:
        with pytest.raises(OpenAIError) as error:
            result = client.models.list()
            if inspect.isawaitable(result):
                await result
        assert "fake-private-credential" not in "".join(traceback.format_exception(error.value))
    finally:
        closed = client.close()
        if inspect.isawaitable(closed):
            await closed


@pytest.mark.parametrize(
    ("asynchronous", "mode"),
    [
        (False, "api_key"),
        (False, "azure_ad_token"),
        (False, "token_provider"),
        (True, "api_key"),
        (True, "azure_ad_token"),
        (True, "token_provider"),
        (True, "async_token_provider"),
    ],
)
async def test_realtime_websocket_authentication_and_refresh(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    tokens = iter(["fake-first-token", "fake-second-token"])

    async def async_token() -> str:
        return next(tokens)

    options: dict[str, Any]
    if mode == "async_token_provider":
        options = {"token_provider": async_token}
    elif mode == "token_provider":
        options = {"token_provider": lambda: next(tokens)}
    else:
        options = {mode: "fake-static-credential"}

    connector = AsyncMock() if asynchronous else Mock()
    monkeypatch.setattr(
        "openai.lib._websocket._WebSocketConnect" if asynchronous else "websockets.sync.client.connect", connector
    )
    provider = azure(**options)
    headers = {"X-Default": "kept", "X-Removed": "remove-me"}
    client = (
        AsyncOpenAI(provider=provider, default_headers=headers)
        if asynchronous
        else OpenAI(provider=provider, default_headers=headers)
    )
    extra_headers = {"X-Removed": omit, "X-Custom": "connection"}
    try:
        for _ in range(2):
            if isinstance(client, AsyncOpenAI):
                async with client.realtime.connect(model="fake-deployment", extra_headers=extra_headers):
                    pass
            else:
                with client.realtime.connect(model="fake-deployment", extra_headers=extra_headers):
                    pass
    finally:
        closed = client.close()
        if inspect.isawaitable(closed):
            await closed

    expected = (
        ["fake-first-token", "fake-second-token"] if mode.endswith("token_provider") else ["fake-static-credential"] * 2
    )
    assert connector.call_count == 2
    for call, credential in zip(connector.call_args_list, expected, strict=True):
        assert call.args == ("wss://azure.example/openai/v1/realtime?model=fake-deployment",)
        handshake_headers = httpx2.Headers(call.kwargs["additional_headers"])
        if mode == "api_key":
            assert handshake_headers["api-key"] == credential
            assert "authorization" not in handshake_headers
        else:
            assert handshake_headers["authorization"] == f"Bearer {credential}"
            assert "api-key" not in handshake_headers
        assert handshake_headers["X-Default"] == "kept"
        assert handshake_headers["X-Custom"] == "connection"
        assert "X-Removed" not in handshake_headers
        assert "Host" not in handshake_headers
        assert credential not in caplog.text
    assert extra_headers == {"X-Removed": omit, "X-Custom": "connection"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "websocket_url",
    ["wss://other.example/openai/v1", "wss://azure.example:444/openai/v1", "ws://azure.example/openai/v1"],
)
async def test_realtime_websocket_rejects_other_origins_before_resolving_tokens(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, websocket_url: str
) -> None:
    def unexpected_token() -> str:
        pytest.fail("A different WebSocket origin must not resolve credentials")

    connector = AsyncMock() if asynchronous else Mock()
    monkeypatch.setattr(
        "openai.lib._websocket._WebSocketConnect" if asynchronous else "websockets.sync.client.connect", connector
    )
    provider = azure(token_provider=unexpected_token)
    client = (
        AsyncOpenAI(provider=provider, websocket_base_url=websocket_url)
        if asynchronous
        else OpenAI(provider=provider, websocket_base_url=websocket_url)
    )
    try:
        with pytest.raises(OpenAIError, match="outside the configured origin"):
            if isinstance(client, AsyncOpenAI):
                async with client.realtime.connect(model="fake-deployment"):
                    pass
            else:
                with client.realtime.connect(model="fake-deployment"):
                    pass
        connector.assert_not_called()
    finally:
        closed = client.close()
        if inspect.isawaitable(closed):
            await closed
