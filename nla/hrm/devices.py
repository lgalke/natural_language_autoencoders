"""CLI defaults: use the GPU + bf16 when there is one (the scripts used to
default to CPU/fp32, silently running 1.5B-param training on CPU)."""

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype() -> str:
    return "bfloat16" if torch.cuda.is_available() else "float32"
