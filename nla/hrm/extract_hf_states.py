"""Extract single-stream hidden states of a standard HF causal LM on the SAME prompt corpus as `stage0_hrm`.

    python -m nla.hrm.extract_hf_states --corpus corpus_v2.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct --layer-index 20 \\
        --output base_qwen15_v2.parquet

The control for the describability ladder (`feature_ceiling`): is Mimir's state less nameable than a standard transformer's on
the same texts? Prompts are rendered with the model's own chat template (user turn + generation prompt, so the same
content-plus-template-tail structure as Mimir's), positions are sampled with the same per-document keyed RNG as `stage0_hrm`
(`_sample_positions`, min position 3, always the last prompt position), and the state is the output of decoder block
`layer-index` (0-based, post-residual: the convention of `nla.datagen.extractors.HFExtractor` and of the released NLAs, e.g. layer
20 of 28 for Qwen2.5-7B). Output columns match `stage0_hrm` where they exist, with a single vector `z` instead of z_L / z_H, so
`feature_ceiling` runs on it unchanged. Token positions refer to THIS model's tokenization, so the document split is shared
(doc_id) but the exact rows are not.
"""

import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config
from nla.hrm.devices import default_device, default_dtype
from nla.hrm.stage0_hrm import _sample_positions


def render(tokenizer, prompt: str) -> list[int]:
    out = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    return list(out["input_ids"] if hasattr(out, "keys") else out)  # BatchEncoding on recent transformers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="JSONL from build_prompt_corpus.py")
    p.add_argument("--base-model", required=True)
    p.add_argument("--layer-index", type=int, required=True)
    p.add_argument("--positions-per-prompt", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(args.device).eval()
    d_model = resolve_text_config(model.config).hidden_size
    layers = resolve_decoder_layers(model)
    assert 0 <= args.layer_index < len(layers), f"layer {args.layer_index} outside 0..{len(layers) - 1}"

    rows = [json.loads(line) for line in open(args.corpus) if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    ids = [render(tokenizer, r["prompt"]) for r in rows]
    keep = sorted((i for i, x in enumerate(ids) if 4 < len(x) <= args.max_prompt_tokens), key=lambda i: len(ids[i]))
    print(f"{len(keep)}/{len(rows)} prompts kept ({len(rows) - len(keep)} too short or longer than {args.max_prompt_tokens} tokens)")

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().float()

    handle = layers[args.layer_index].register_forward_hook(hook)
    schema = pa.schema([("z", pa.list_(pa.float32(), d_model)), ("position", pa.int64()), ("prompt_len", pa.int64()),
                        ("is_last_prompt_pos", pa.bool_()), ("dataset", pa.string()), ("world", pa.string()),
                        ("doc_id", pa.string()), ("prompt_ids", pa.list_(pa.int64()))])
    n_rows = 0
    try:
        with pq.ParquetWriter(args.output, schema) as writer:
            for s in tqdm(range(0, len(keep), args.batch_size), desc="batches"):
                chunk = keep[s:s + args.batch_size]
                width = max(len(ids[i]) for i in chunk)
                batch = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), width), dtype=torch.long)
                for b, i in enumerate(chunk):  # right padding: causal model, real tokens never see the padding
                    batch[b, :len(ids[i])] = torch.tensor(ids[i])
                    attn[b, :len(ids[i])] = 1
                with torch.no_grad():
                    model.model(input_ids=batch.to(args.device), attention_mask=attn.to(args.device), use_cache=False)
                out: dict[str, list] = {k: [] for k in schema.names}
                for b, i in enumerate(chunk):
                    length = len(ids[i])
                    for pos in _sample_positions(length, args.positions_per_prompt, rows[i]["doc_id"], args.seed):
                        out["z"].append(captured["h"][b, pos].cpu().tolist())
                        out["position"].append(pos)
                        out["prompt_len"].append(length)
                        out["is_last_prompt_pos"].append(pos == length - 1)
                        out["dataset"].append(rows[i]["dataset"])
                        out["world"].append(rows[i].get("world", rows[i]["doc_id"]))
                        out["doc_id"].append(rows[i]["doc_id"])
                        out["prompt_ids"].append(ids[i])
                if out["doc_id"]:
                    writer.write_table(pa.Table.from_pydict(out, schema=schema))
                    n_rows += len(out["doc_id"])
    finally:
        handle.remove()
    print(f"wrote {n_rows} rows -> {args.output}")


if __name__ == "__main__":
    main()
