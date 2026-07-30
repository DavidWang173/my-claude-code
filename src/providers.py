"""Provider-neutral contracts, errors, and registration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from .models import ModelRequest, ModelResponse, ModelStreamChunk


class ModelProvider(Protocol):
    """Interface implemented by remote and local model adapters."""

    @property
    def name(self) -> str: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]: ...

    async def aclose(self) -> None: ...


class ProviderError(RuntimeError):
    """Base class for safe, provider-neutral failures."""


class ProviderTimeoutError(ProviderError):
    pass


class ProviderConfigurationError(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderServerError(ProviderError):
    pass


class ProviderConnectionError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass


class ProviderInvalidResponseError(ProviderError):
    pass


class ProviderNotFoundError(LookupError):
    pass


class ProviderRegistry:
    """Per-application provider registry; deliberately not a singleton."""

    def __init__(self, providers: Iterable[ModelProvider] = ()) -> None:
        self._providers: dict[str, ModelProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ModelProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"provider already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> ModelProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ProviderNotFoundError(f"provider is not configured: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))
