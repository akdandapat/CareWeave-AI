"""LLM provider abstraction.

Two implementations behind one interface:

* ``StubProvider`` -- deterministic, offline, no API key. Uses lexical rules for
  extraction and templated composition for generation.
* ``AnthropicProvider`` -- real model calls.

This is not just convenience. It buys three things that matter in review:

1. **The full pipeline runs for anyone who clones the repo.** No key, no cost,
   no rate limit, same output every time.
2. **The LLM's contribution is measurable.** Every evaluation runs against both
   providers, so "what did the language model actually add?" has a number rather
   than an assertion.
3. **The system degrades rather than fails.** If the API is unavailable at
   runtime, the stub answers. In a healthcare workflow a deterministic, narrower
   answer is a far better failure mode than an outage.

The generation contract is deliberately narrow: the provider receives an
already-selected action and a fixed evidence set, and produces prose. It never
chooses what to do.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from careweave.config import SETTINGS


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    #: populated for structured calls
    parsed: dict[str, Any] | None = None
    error: str | None = None


class LLMProvider(Protocol):
    name: str

    def complete(self, *, system: str, user: str, max_tokens: int = 1000) -> LLMResult: ...

    def complete_json(
        self, *, system: str, user: str, schema_hint: str, max_tokens: int = 1000
    ) -> LLMResult: ...


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------


class StubProvider:
    """Deterministic offline provider.

    Structured calls are answered by the caller-supplied fallback (the extractor
    passes its own rule-based result through). Free-text calls return the
    templated draft the caller has already assembled. In both cases the stub is
    a pass-through that keeps the pipeline shape identical to the API path, so
    the graph, the gate and the ledger behave the same either way.
    """

    name = "stub"

    def __init__(self, model: str = "deterministic-v1") -> None:
        self.model = model

    def complete(self, *, system: str, user: str, max_tokens: int = 1000) -> LLMResult:
        # The caller passes its deterministic draft as the final line of `user`
        # under a DRAFT: marker; the stub echoes it.
        m = re.search(r"DRAFT:\n(.*)\Z", user, flags=re.S)
        text = m.group(1).strip() if m else user.strip()
        return LLMResult(text=text, provider=self.name, model=self.model)

    def complete_json(
        self, *, system: str, user: str, schema_hint: str, max_tokens: int = 1000
    ) -> LLMResult:
        m = re.search(r"FALLBACK:\n(\{.*\})\s*\Z", user, flags=re.S)
        parsed = json.loads(m.group(1)) if m else {}
        return LLMResult(
            text=json.dumps(parsed), provider=self.name, model=self.model, parsed=parsed
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or SETTINGS.llm.model
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "anthropic package not installed; set CW_LLM_PROVIDER=stub"
                ) from exc
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def complete(self, *, system: str, user: str, max_tokens: int = 1000) -> LLMResult:
        try:
            client = self._client_or_raise()
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=SETTINGS.llm.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return LLMResult(text=text.strip(), provider=self.name, model=self.model)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the graph
            return LLMResult(text="", provider=self.name, model=self.model, error=str(exc))

    def complete_json(
        self, *, system: str, user: str, schema_hint: str, max_tokens: int = 1000
    ) -> LLMResult:
        sys_prompt = (
            f"{system}\n\nRespond with a single JSON object and nothing else. "
            f"No prose, no markdown fences.\nSchema:\n{schema_hint}"
        )
        result = self.complete(system=sys_prompt, user=user, max_tokens=max_tokens)
        if result.error:
            return result
        raw = re.sub(r"^```(?:json)?|```$", "", result.text.strip(), flags=re.M).strip()
        try:
            result.parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            result.error = f"json parse failed: {exc}"
            result.parsed = None
        return result


# ---------------------------------------------------------------------------


_PROVIDER: LLMProvider | None = None


def get_provider(force: str | None = None) -> LLMProvider:
    global _PROVIDER
    name = force or SETTINGS.llm.provider
    if force is not None or _PROVIDER is None:
        provider: LLMProvider = (
            AnthropicProvider() if name == "anthropic" else StubProvider()
        )
        if force is not None:
            return provider
        _PROVIDER = provider
    return _PROVIDER
