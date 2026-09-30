"""Completion provider for any OpenAI-compatible chat/completions endpoint
(local/hosted GLM, OpenRouter, vLLM, ...), via httpx — no vendor SDK.

Example (UCloud-hosted GLM-5.3, matching the opencode config):
    export UCLOUD_API_KEY=...        # or put UCLOUD_API_KEY=... in ./.env
    python -m nla.hrm.explain ... \\
        --provider-cls nla.hrm.openai_compat.OpenAICompatProvider \\
        --provider-kwargs '{"base_url": "https://ai.cloud.sdu.dk/v1",
                            "model": "zai-org/GLM-5.3", "api_key_env": "UCLOUD_API_KEY",
                            "reasoning_effort": "low", "max_tokens": 2000}'

Key lookup: `api_key` kwarg, then the env var named by `api_key_env`, then a
`NAME=value` line in ./.env (minimal parser, no python-dotenv). Never printed.

Reasoning models spend max_tokens on hidden reasoning first — give them
headroom (or a low reasoning_effort) or the visible answer gets cut off before
the closing </analysis> tag and the row is dropped.
"""

import asyncio
import os
from pathlib import Path

import httpx

from nla.datagen.providers import CompletionProvider


def _key_from_env_file(name: str, path: str = ".env") -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith(name) and "=" in line and line.split("=", 1)[0].strip() == name:
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


class OpenAICompatProvider(CompletionProvider):
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY",
                 api_key: str | None = None, max_tokens: int = 2000, temperature: float = 1.0,
                 reasoning_effort: str | None = None, concurrency: int = 16, max_retries: int = 5,
                 extra_body: dict | None = None):
        key = api_key or os.environ.get(api_key_env) or _key_from_env_file(api_key_env)
        assert key, f"no API key: set ${api_key_env}, put it in ./.env, or pass api_key"
        self._key = key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.concurrency, self.max_retries = concurrency, max_retries
        self.extra = dict(extra_body or {})
        if reasoning_effort:
            self.extra["reasoning_effort"] = reasoning_effort

    async def _one(self, client: httpx.AsyncClient, sem: asyncio.Semaphore, prompt: str) -> str | None:
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}], **self.extra}
        async with sem:
            for attempt in range(self.max_retries):
                try:
                    r = await client.post(self.url, json=body, headers={"Authorization": f"Bearer {self._key}"})
                except httpx.TransportError:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if r.status_code in (401, 403):
                    raise RuntimeError(f"auth failed ({r.status_code}) at {self.url}: {r.text[:200]}")
                if r.status_code == 429 or r.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"error {r.status_code} at {self.url}: {r.text[:300]}")
                choices = r.json().get("choices") or []
                text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
                return text or None
        return None  # retries exhausted -> row dropped by the caller

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run():
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient(timeout=300) as client:
                return await asyncio.gather(*(self._one(client, sem, p) for p in prompts))
        out = asyncio.run(_run())
        n_none = sum(o is None for o in out)
        if n_none:
            print(f"  [OpenAICompatProvider] {n_none}/{len(prompts)} completions failed/empty")
        return out


class UCloudGLMProvider(OpenAICompatProvider):
    """UCloud-hosted GLM with the defaults we want for explanations: LOW
    reasoning effort (the server default is "max", which is slow and can eat
    the whole token budget before the answer), key from $UCLOUD_API_KEY / ./.env.
    Override anything via --provider-kwargs, e.g. '{"model": "zai-org/GLM-5.3-Flash"}'."""

    def __init__(self, base_url: str = "https://ai.cloud.sdu.dk/v1", model: str = "zai-org/GLM-5.3",
                 api_key_env: str = "UCLOUD_API_KEY", reasoning_effort: str | None = "low",
                 max_tokens: int = 2000, **kwargs):
        super().__init__(base_url=base_url, model=model, api_key_env=api_key_env,
                         reasoning_effort=reasoning_effort, max_tokens=max_tokens, **kwargs)
