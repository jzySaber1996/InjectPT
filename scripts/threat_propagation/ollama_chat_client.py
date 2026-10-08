#!/usr/bin/env python3
"""Small Ollama /api/chat client used by local threat-propagation scripts."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
DEFAULT_OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
DEFAULT_OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")
DEFAULT_OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "false").strip().lower() or "false"


class OllamaRequestError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def strip_thinking_blocks(text: str) -> str:
    """Remove Qwen-style thinking traces before downstream JSON parsing."""
    stripped = re.sub(r"(?is)<think>.*?</think>", "", text or "")
    return stripped.strip()


def rewrite_prompt_for_ollama(prompt: str, *, model_label: str = "local Ollama/Qwen3 model") -> str:
    """Keep existing prompts but avoid provider-specific model identity drift."""
    return (
        prompt.replace("DeepSeek-guided", "Ollama/Qwen3-guided")
        .replace("DeepSeek-generated", "Ollama/Qwen3-generated")
        .replace("DeepSeek", model_label)
        .replace("deepseek", model_label)
    )



def rewrite_object_for_ollama(value: Any, *, model_label: str = "local Ollama/Qwen3 model") -> Any:
    """Recursively rewrite model-name references inside prompt payload objects."""
    if isinstance(value, str):
        return rewrite_prompt_for_ollama(value, model_label=model_label)
    if isinstance(value, list):
        return [rewrite_object_for_ollama(item, model_label=model_label) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_object_for_ollama(item, model_label=model_label) for key, item in value.items()}
    return value

def parse_ollama_options(raw_options: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for raw_option in raw_options:
        key, separator, value = raw_option.partition("=")
        key = key.strip()
        if not separator or not key:
            raise SystemExit(f"--ollama-option must use key=value syntax: {raw_option!r}")
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        options[key] = parsed_value
    return options


def parse_ollama_think(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise SystemExit("--ollama-think must be one of: auto, true, false")


class OllamaClient:
    def __init__(
        self,
        *,
        model: str,
        api_url: str,
        retries: int,
        retry_seconds: float,
        timeout_seconds: float,
        keep_alive: str,
        think: bool | None,
        use_json_format: bool,
        options: dict[str, Any] | None = None,
    ):
        if not model.strip():
            raise ValueError("Ollama model is required. Pass --model or set OLLAMA_MODEL.")
        self.model = model.strip()
        self.api_url = api_url.strip()
        self.retries = max(0, retries)
        self.retry_seconds = max(0.0, retry_seconds)
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.keep_alive = keep_alive.strip()
        self.think = think
        self.use_json_format = use_json_format
        self.options = dict(options or {})

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        request_label: str = "chat",
    ) -> str:
        prompt_chars = len(system_prompt) + len(user_prompt)
        payload = self._build_payload(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            include_optional_fields=True,
        )
        can_fallback_optional_fields = "think" in payload or "format" in payload
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                data = self._post(payload)
                return self._extract_content(data)
            except OllamaRequestError as exc:
                last_error = exc
                if exc.status == 400 and can_fallback_optional_fields:
                    payload = self._build_payload(
                        system_prompt,
                        user_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        include_optional_fields=False,
                    )
                    can_fallback_optional_fields = False
                    print(
                        f"[ollama-warning] label={request_label} endpoint rejected optional "
                        "chat fields; retrying without think/format",
                        file=sys.stderr,
                    )
                    continue
            except Exception as exc:  # pragma: no cover - defensive CLI boundary
                last_error = exc

            if attempt >= self.retries:
                break
            delay = self.retry_seconds * (2**attempt)
            print(
                f"[ollama-retry] label={request_label} error={last_error} "
                f"prompt_chars={prompt_chars} attempt={attempt + 1}/{self.retries + 1}; "
                f"sleep={delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)

        if last_error is not None:
            raise RuntimeError(
                f"Ollama request failed for {request_label}; model={self.model}; "
                f"api_url={self.api_url}; prompt_chars={prompt_chars}; error={last_error}"
            ) from last_error
        raise RuntimeError("Ollama request failed without an error")

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        include_optional_fields: bool,
    ) -> dict[str, Any]:
        options = dict(self.options)
        options["temperature"] = temperature
        options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if include_optional_fields:
            if self.think is not None:
                payload["think"] = self.think
            if self.use_json_format:
                payload["format"] = "json"
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = body.strip() or str(exc)
            raise OllamaRequestError(f"HTTP {exc.code}; {detail}", status=exc.code) from exc
        except urllib.error.URLError as exc:
            raise OllamaRequestError(
                f"{exc}. Make sure `ollama serve` is running, "
                "the model is pulled, and --api-url points to /api/chat."
            ) from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaRequestError(f"Invalid JSON response from Ollama: {body[:500]!r}") from exc
        if not isinstance(data, dict):
            raise OllamaRequestError(f"Expected JSON object from Ollama: {body[:500]!r}")
        return data

    def _extract_content(self, data: dict[str, Any]) -> str:
        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        if not content and isinstance(data.get("response"), str):
            content = data["response"]
        if not isinstance(content, str) or not content.strip():
            raise OllamaRequestError(f"Ollama response missing message.content: {json.dumps(data)[:500]}")
        return strip_thinking_blocks(content)
