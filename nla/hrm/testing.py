"""Test-only helpers — a `CompletionProvider` that needs no network/API key,
for the smoke pipeline (docs/hrm.md "Verification") and pytest. NOT imported
by any training/datagen path; only ever named via `--provider-cls` on the CLI
or imported directly from tests.
"""

import hashlib

from nla.datagen.providers import CompletionProvider


class FakeCompletionProvider(CompletionProvider):
    """Deterministic-but-varying fake explanations, formatted exactly like the
    real instruction prompt expects (`<analysis>...</analysis>`, 2+ `\\n\\n`-
    separated features) so `_extract_and_clean` in stage2_api_explain.py
    accepts them. Two calls with the SAME prompt text return DIFFERENT text
    (keyed by a call counter) — needed for `--samples-per-row>1`, where the
    same prompt is sent twice and must come back as two distinct fields.
    """

    def __init__(self, **_kwargs):
        self._n_calls = 0

    def complete(self, prompts: list[str]) -> list[str | None]:
        out = []
        for prompt in prompts:
            h = hashlib.sha256(f"{self._n_calls}|{prompt}".encode()).hexdigest()[:8]
            self._n_calls += 1
            out.append(
                f"<analysis>\nfake feature A ({h})\n\nfake feature B ({h})\n</analysis>"
            )
        return out
