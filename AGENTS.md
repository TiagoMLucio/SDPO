# SDPO fork (rebase-validation repro)

Fork of [lasgroup/SDPO](https://github.com/lasgroup/SDPO) (the SDPO paper's code release),
used for ONE purpose: validating that our rebased verl reproduces the paper's rich-feedback
result on LCBv6. Orchestration lives in the sibling `../agentic_sdpo_training` checkout.

## What is different from upstream

- `verl/` (the vendored package) is a **file copy of TiagoMLucio/verl** (rebased on recent
  upstream verl: sync/TransferQueue trainer, per-segment SDPO teacher, langfuse tracing),
  swapped in commit `0313b17`. It is NOT a git fork inside this repo; to update, re-rsync from
  `agentic_sdpo_training/uni-agent/verl/verl/` and restore the three kept files below.
- Kept from upstream SDPO (do not overwrite in a re-sync):
  `verl/utils/reward_score/feedback/` (feedback-emitting rewards, loaded by file path),
  `verl/trainer/config/user.yaml` (experiment config; paths are env-overridable via
  SDPO_DIR / SDPO_LOG_DIR / SDPO_CKPT_DIR), `verl/trainer/config/sdpo.yaml` (defaults include `user`).
- Everything else (experiments/, data/, training/, datasets/) is upstream.

## Do NOT

- Edit files under `verl/` here directly: make the change in the verl fork and re-sync.
  Exception: the three kept experiment files above.
- Use this repo's requirements/Dockerfile for the swapped verl: they target the OLD verl
  (torch 2.5.1 / vllm 0.8.4). The working env is defined in
  `agentic_sdpo_training/modal/sdpo_repro.py` (vllm v0.12 base + fork requirements).

## Running

- Launch (Modal, 4xH200): `cd ../agentic_sdpo_training/modal && modal run --detach sdpo_repro.py::train --steps 10 --exp-name sdpo-sanity-10`.
  This applies the exact overrides of `experiments/rich_feedback/run_sdpo.sh` through
  `verl.trainer.main_ppo_sync`, plus a step pin (`trainer.total_training_steps`).
- Data prep (already done for lcb_v6): `data/load_dataset.py` -> `data/split_tests.py` -> `data/preprocess.py` (see `data/README.md`); parquets in `datasets/lcb_v6/`.
- Paper reference: 80 steps, batch 32, n=8, alpha=1.0, topk=20, Qwen3-8B, 4xGH200 (~6h).
