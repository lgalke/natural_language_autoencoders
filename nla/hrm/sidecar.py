"""Sidecar YAML for HRM two-stream datasets/models — `{parquet}.nla_meta.yaml`.

Parallel to `nla/datagen/sidecar.py` but for a schema that carries TWO
activation streams (z_L, z_H) per row instead of one `activation_vector`.
Reuses the shared path/token helpers from `nla.schema` and the `Storage`
abstraction from `nla.storage` so the rest of the datagen plumbing
(`nla/datagen/injection_tokens.py`, `nla/schema.py:compute_canonical_neighbors`)
works unchanged against this sidecar's `tokens` / `prompt_templates` fields.
"""

from __future__ import annotations

import dataclasses
import datetime
import subprocess
from dataclasses import asdict, dataclass, field

import yaml

from nla.storage import Storage, dataset_sidecar_path

SCHEMA_VERSION = 1
KIND = "nla_hrm_dataset"

_TRAINING_STAGES = {"av_sft", "ar_sft", "rl"}


@dataclass
class HrmTokenMeta:
    """TWO injection markers (not one — nla.schema.NLATokenMeta doesn't fit;
    see nla/hrm/injection.py module docstring), plus the critic suffix (which
    doesn't need a marker at all, same convention as the rest of NLA)."""
    injection_char_L: str
    injection_token_id_L: int
    injection_left_neighbor_id_L: int
    injection_right_neighbor_id_L: int
    injection_char_H: str
    injection_token_id_H: int
    injection_left_neighbor_id_H: int
    injection_right_neighbor_id_H: int
    critic_suffix_ids: list[int] | None = None


@dataclass
class HrmExtractionMeta:
    base_model: str  # Mimir checkpoint the streams were extracted from
    d_model: int
    hook_site: str  # human-readable, e.g. "H_in@2 (z_L=L_out@2.3, z_H=H_out@1)"
    L_cycles: int
    H_cycles: int
    norm: str  # datagen always writes "none" — RAW z_L/z_H, see CLAUDE.md
    corpus: str  # description / path of the prompt corpus used
    positions_per_prompt: int
    streams: list[str] = field(default_factory=lambda: ["z_L", "z_H"])


@dataclass
class HrmDatasetMeta:
    dataset_id: str
    stage: str  # base | av_sft | ar_sft | rl | eval_iid | eval_ood
    row_count: int
    extraction: HrmExtractionMeta
    kind: str = KIND
    schema_version: int = SCHEMA_VERSION
    verbalizer_model: str | None = None  # Qwen2.5-1.5B-Instruct, once fixed
    tokens: HrmTokenMeta | None = None  # injection char/id/neighbors — verbalizer tokenizer
    prompt_templates: dict[str, str] = field(default_factory=dict)  # {"actor":..., "critic":...}
    norm_stats_path: str | None = None
    norm_stats_hash: str | None = None
    api_summary: dict[str, object] | None = None  # model/max_tokens/temperature/instruction, from explain.py
    keep_debug_metadata: bool = True
    parent_datasets: list[str] = field(default_factory=list)
    created_at: str = ""
    created_by: str = ""
    git_commit: str = ""


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def serialize_sidecar(meta: HrmDatasetMeta) -> str:
    if meta.stage in _TRAINING_STAGES:
        assert meta.tokens is not None, (
            f"stage={meta.stage!r} requires HrmTokenMeta — training reads the "
            f"injection token IDs from it."
        )
        assert meta.prompt_templates.get("actor"), (
            f"stage={meta.stage!r} requires prompt_templates['actor']"
        )
        if meta.stage == "ar_sft":
            assert meta.tokens.critic_suffix_ids is not None, (
                "stage=ar_sft requires critic_suffix_ids for last-token extraction"
            )
            assert meta.prompt_templates.get("critic"), (
                "stage=ar_sft requires prompt_templates['critic']"
            )
    if not meta.created_at:
        meta.created_at = datetime.datetime.now(tz=datetime.UTC).isoformat()
    if not meta.git_commit:
        meta.git_commit = _git_commit()
    d = {k: v for k, v in asdict(meta).items() if v is not None}
    return yaml.safe_dump(d, sort_keys=False)


def deserialize_sidecar(text: str) -> HrmDatasetMeta:
    d = yaml.safe_load(text)
    assert d["kind"] == KIND, f"not an HRM dataset sidecar: kind={d['kind']!r}"
    assert d["schema_version"] == SCHEMA_VERSION, (
        f"sidecar schema version {d['schema_version']} != expected {SCHEMA_VERSION}"
    )
    d["extraction"] = HrmExtractionMeta(**d["extraction"])
    if d.get("tokens") is not None:
        d["tokens"] = HrmTokenMeta(**d["tokens"])
    known = {f.name for f in dataclasses.fields(HrmDatasetMeta)}
    return HrmDatasetMeta(**{k: v for k, v in d.items() if k in known})


def write_sidecar(storage: Storage, parquet_path: str, meta: HrmDatasetMeta) -> None:
    storage.write_text(dataset_sidecar_path(parquet_path), serialize_sidecar(meta))


def read_sidecar(storage: Storage, parquet_path: str) -> HrmDatasetMeta:
    return deserialize_sidecar(storage.read_text(dataset_sidecar_path(parquet_path)))
