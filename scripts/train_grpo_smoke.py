"""GRPO smoke test (TRL + LoRA). Purpose: confirm the reward is learnable and
does not collapse into a constant policy. It is NOT a recipe for a good model.

What to watch in the logs:
  rewards/monitor_creature   should rise
  rewards/monitor_prop       must stay high. If it falls while creature rises,
                             the policy is learning "always swerve" and the
                             reward or the kind mix needs changing.
  rewards/monitor_rock       should rise or stay high
  rewards/format_ok          should be ~1.0 within a few steps

RunPod, one 48GB+ GPU (A6000 / L40S / A100):
    pip install "trl>=0.19" peft accelerate datasets
    pip install -e .            # this repo; pulls the harvest engine from GitHub
    python scripts/train_grpo_smoke.py --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 200
    # then the CaML standard test model (gated, needs HF_TOKEN):
    python scripts/train_grpo_smoke.py --model meta-llama/Llama-3.1-8B-Instruct --max-steps 300

NOTE: written against the TRL GRPOTrainer API and not yet executed on a GPU.
Expect to adjust batch sizes for your card.
"""

from __future__ import annotations

import argparse
import json
import logging

from datasets import Dataset

from harvest_rush_train.generate import generate_examples
from harvest_rush_train.reward import MODES, parse_choice, score_choice

log = logging.getLogger("train_grpo_smoke")


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"] if completion else ""


def build_dataset(n: int, split: str, seed: int, mode: str) -> Dataset:
    rows = generate_examples(n, split, seed, mode)
    return Dataset.from_list([{"prompt": r["prompt"],
                               "info": json.dumps(r["info"])} for r in rows])


class KindTracker:
    """Per-kind running view of what the policy is doing, logged every
    `every` reward calls. GRPOTrainer's own per-kind monitors are averaged
    over logging windows and read NaN whenever one batch lacks a kind, so
    they cannot show whether animals and props move independently."""

    def __init__(self, every: int = 20):
        self.every, self.calls = every, 0
        self.reset()

    def reset(self):
        self.n = {k: 0 for k in ("creature", "prop", "rock")}
        self.cont = dict(self.n)
        self.rew = {k: 0.0 for k in self.n}

    def add(self, kind, choice, reward):
        if kind in self.n:
            self.n[kind] += 1
            self.cont[kind] += int(choice == "continue")
            self.rew[kind] += reward

    def tick(self):
        self.calls += 1
        if self.calls % self.every == 0:
            parts = [f"{k}: continue={self.cont[k] / self.n[k]:.2f} reward={self.rew[k] / self.n[k]:.2f} n={self.n[k]}"
                     for k in self.n if self.n[k]]
            log.info("KIND_TRACK call=%d | %s", self.calls, " | ".join(parts))
            self.reset()


def make_reward_funcs(mode: str):
    tracker = KindTracker()

    def choice_reward(prompts, completions, info, **_):
        out = []
        for c, i in zip(completions, info):
            i = json.loads(i)
            choice = parse_choice(_text(c))
            r = score_choice(choice, i, mode)
            tracker.add(i["kind"], choice, r)
            out.append(r)
        tracker.tick()
        return out

    def format_ok(prompts, completions, info, **_):
        return [1.0 if parse_choice(_text(c)) in json.loads(i)["options"] else 0.0
                for c, i in zip(completions, info)]

    def monitor(kind: str):
        # per-kind view of the SAME reward; None = not applicable (TRL skips it)
        def fn(prompts, completions, info, **_):
            out = []
            for c, i in zip(completions, info):
                i = json.loads(i)
                out.append(score_choice(parse_choice(_text(c)), i, mode)
                           if i["kind"] == kind else None)
            return out
        fn.__name__ = f"monitor_{kind}"
        return fn

    funcs = [choice_reward, format_ok, monitor("creature"), monitor("prop"),
             monitor("rock")]
    weights = [1.0, 0.0, 0.0, 0.0, 0.0]
    return funcs, weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", choices=MODES, default="control_consistent")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL coefficient against the reference policy (0 = none)")
    ap.add_argument("--no-scale-rewards", action="store_true",
                    help="do not divide group advantages by the group std, so reward "
                         "magnitudes (e.g. harm_averse's -1.0) keep their meaning")
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--output-dir", default="runs/grpo_smoke")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    # heavy imports after arg parsing so --help is instant
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    log.info("args: %s", vars(args))
    train = build_dataset(args.n_train, "train", args.seed, args.mode)
    evals = build_dataset(args.n_eval, "eval", args.seed, args.mode)
    log.info("datasets ready: train=%d eval=%d", len(train), len(evals))

    funcs, weights = make_reward_funcs(args.mode)
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=1024,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        temperature=1.0,
        beta=args.beta,
        scale_rewards=not args.no_scale_rewards,
        reward_weights=weights,
        logging_steps=5,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        per_device_eval_batch_size=args.per_device_batch,
        bf16=True,
        use_vllm=args.use_vllm,
        report_to=args.report_to,
        seed=args.seed,
        log_completions=True,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=funcs,
        args=cfg,
        train_dataset=train,
        eval_dataset=evals,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                               target_modules="all-linear",
                               task_type="CAUSAL_LM"),
    )
    log.info("starting GRPO: model=%s steps=%d", args.model, args.max_steps)
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    log.info("saved adapter to %s/final", args.output_dir)


if __name__ == "__main__":
    main()
