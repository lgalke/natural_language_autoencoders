"""HRM verbalizer extension — natural-language autoencoder for Mimir's L/H recurrent states.

See docs/hrm.md for the design and CLAUDE.md's "HRM extension" section for the
invariants. This package is intentionally decoupled from nla/train_actor.py and
the Miles/SGLang stack (see docs/hrm.md "Why not Miles"): it's a single-process
trainer built around HuggingFace `transformers` + `peft`, reusing only the
datagen/sidecar/injection machinery from the rest of `nla/`.
"""
