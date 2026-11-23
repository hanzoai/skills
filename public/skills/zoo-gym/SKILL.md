---
name: zoo-gym
description: Three training methods from Zoo Gym — Training-Free GRPO (RLHF with no value network), BitDelta per-user personalisation, and DeltaSoup Byzantine-robust community learning. Use when training or personalising a ZenLM model, or choosing an RLHF method.
---

# Zoo Gym training methods

Zoo Gym is the training platform for ZenLM models, maintained by Zoo Labs
Foundation (zoo.ngo, github.com/zooai/gym) — a separate organisation from
Hanzo AI. These are its three methods, each with its own file here.

- `training-free-grpo.md` — RLHF without a value network. The default for
  RLHF in Zoo Gym, and the one to reach for first.
- `bitdelta.md` — per-user personalisation without LoRA, at about a tenth of
  the memory. Specified by ZIP-7.
- `deltasoup.md` — community learning that survives Byzantine contributors,
  with a reward for honest ones.

`INDEX.md` describes all three in more detail, including when each applies.
