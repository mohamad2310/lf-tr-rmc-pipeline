from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol


class LLMClientError(RuntimeError):
    """Base error for provider-layer failures."""


class MissingPackageError(LLMClientError):
    """Raised when the required SDK is not installed."""


class MissingAPIKeyError(LLMClientError):
    """Raised when no usable API key is available."""


class EmptyResponseError(LLMClientError):
    """Raised when the provider returned no plain-text content."""


class PromptTooLargeError(LLMClientError):
    """Raised when the provider rejects the prompt because it exceeds context limits."""


class BaseLLMClient(Protocol):
    provider_name: str
    model_name: str

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        ...


def _require_api_key(explicit_key: Optional[str], env_var: str) -> str:
    api_key = (explicit_key or os.getenv(env_var) or "").strip()
    if not api_key:
        raise MissingAPIKeyError(f"Missing API key. Set {env_var} or pass api_key explicitly.")
    return api_key


def _normalize_text_output(parts: list[str], provider_name: str) -> str:
    text = "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    if not text:
        raise EmptyResponseError(f"{provider_name} returned an empty text response.")
    return text


def _is_context_length_error(exc: Exception) -> bool:
    message = str(exc).lower()
    code = str(getattr(exc, "code", "")).lower()
    body = str(getattr(exc, "body", "")).lower()
    details = " ".join([message, code, body])
    return any(
        token in details
        for token in (
            "context_length_exceeded",
            "maximum context length",
            "prompt is too long",
            "input is too long",
            "too many tokens",
            "context window",
        )
    )


@dataclass
class OpenAIClient:
    model_name: str
    api_key: Optional[str] = None
    provider_name: str = "openai"

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingPackageError("Missing package: openai. Install it with `pip install openai`.") from exc

        client = OpenAI(api_key=_require_api_key(self.api_key, "OPENAI_API_KEY"))
        input_items: list[dict[str, object]] = []
        if system_prompt:
            input_items.append({"role": "system", "content": [{"type": "input_text", "text": system_prompt}]})
        input_items.append({"role": "user", "content": [{"type": "input_text", "text": prompt}]})

        try:
            response = client.responses.create(model=self.model_name, input=input_items)
        except Exception as exc:
            if _is_context_length_error(exc):
                raise PromptTooLargeError(f"OpenAI rejected the prompt as too large: {exc}") from exc
            raise
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
                elif hasattr(text_value, "value") and isinstance(text_value.value, str):
                    parts.append(text_value.value)
        return _normalize_text_output(parts, "OpenAI")


@dataclass
class AnthropicClient:
    model_name: str
    api_key: Optional[str] = None
    provider_name: str = "anthropic"

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise MissingPackageError("Missing package: anthropic. Install it with `pip install anthropic`.") from exc

        client = anthropic.Anthropic(api_key=_require_api_key(self.api_key, "ANTHROPIC_API_KEY"))
        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=8192,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            if _is_context_length_error(exc):
                raise PromptTooLargeError(f"Anthropic rejected the prompt as too large: {exc}") from exc
            raise
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return _normalize_text_output(parts, "Anthropic")




def make_llm_client(provider: str, model: str, api_key: Optional[str] = None) -> BaseLLMClient:
    normalized_provider = provider.lower().strip()
    if normalized_provider == "openai":
        return OpenAIClient(model_name=model, api_key=api_key)
    if normalized_provider in {"anthropic", "claude"}:
        return AnthropicClient(model_name=model, api_key=api_key)
    raise ValueError(f"Unsupported LLM provider: {provider}")
