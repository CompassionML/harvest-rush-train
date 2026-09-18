"""Supervised warm start (TRL SFTTrainer + LoRA) on the preferred answers.

Why this exists. GRPO from a small model that starts out swerving at
everything has an exploration problem: on animal prompts nearly every sampled
group is unanimous (no gradient), and whichever kind does give signal drags the
global tendency to "continue" up or down for every kind. Measured on
Qwen2.5-1.5B: control_consistent raised continue on props AND animals;
harm_averse with unscaled advantages drove continue to zero everywhere. The
dataset carries a programmatic `answer` column, so the discrimination can be
taught directly, and GRPO (if wanted) can start from a policy that already
tells a leaf pile from a goat.

    python scripts/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --n-train 2000 --output-dir runs/sft_qwen
    python scripts/eval_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/sft_qwen/final --baseline runs/eval_qwen_base.json

Loss is on the completion only: one line of JSON, {"choice": "<answer>"}.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

from datasets import Dataset

from harvest_rush_train.generate import generate_examples
from harvest_rush_train.reward import MODES

log = logging.getLogger("train_sft")


def build(n: int, split: str, seed: int, mode: str) -> Dataset:
    rows = generate_examples(n, split, seed, mode)
    log.info("%s split: %d rows, answers %s, kinds %s", split, len(rows),
             dict(Counter(r["answer"] for r in rows)),
             dict(Counter(r["info"]["kind"] for r in rows)))
    return Dataset.from_list([
        {"prompt": r["prompt"],
         "completion": [{"role": "assistant",
                         "content": json.dumps({"choice": r["answer"]})}]}
        for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", choices=MODES, default="control_consistent")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--output-dir", default="runs/sft")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("args: %s", vars(args))

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    train = build(args.n_train, "train", args.seed, args.mode)
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_length=1536,
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=train,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    log.info("starting SFT: model=%s rows=%d epochs=%s", args.model, len(train), args.epochs)
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    log.info("saved adapter to %s/final", args.output_dir)


if __name__ == "__main__":
    main()
