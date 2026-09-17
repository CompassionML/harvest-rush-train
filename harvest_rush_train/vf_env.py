"""verifiers entry point (Prime Intellect Environments Hub convention).

    import verifiers as vf
    env = vf.load_environment("harvest_rush_train", mode="control_consistent")

or directly:  from harvest_rush_train import load_environment
"""

from __future__ import annotations

import json
import logging

import verifiers as vf
from datasets import Dataset

from .generate import generate_examples
from .reward import AVOID, MODES, parse_choice, score_choice

log = logging.getLogger("harvest_rush_train.vf_env")


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    for msg in reversed(completion or []):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):  # content parts
                content = "".join(p.get("text", "") if isinstance(p, dict)
                                  else str(getattr(p, "text", "")) for p in content)
            return content or ""
    return ""


def _info(info) -> dict:
    return json.loads(info) if isinstance(info, str) else info


def _to_dataset(rows: list[dict]) -> Dataset:
    # info is stored as a JSON string: HF datasets would otherwise coerce the
    # None/int cost fields into a struct with lossy types.
    return Dataset.from_list([
        {"prompt": r["prompt"], "answer": r["answer"],
         "info": json.dumps(r["info"])} for r in rows])


def load_environment(mode: str = "control_consistent",
                     num_train_examples: int = 2000,
                     num_eval_examples: int = 300,
                     seed: int = 0,
                     conditions: list[str] | None = None,
                     **kwargs) -> vf.Environment:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    log.info("building harvest_rush_train env: mode=%s train=%d eval=%d seed=%d",
             mode, num_train_examples, num_eval_examples, seed)
    train = generate_examples(num_train_examples, "train", seed, mode,
                              conditions=conditions)
    evals = generate_examples(num_eval_examples, "eval", seed, mode,
                              conditions=conditions)

    def choice_reward(completion, info, **_) -> float:
        return score_choice(parse_choice(_completion_text(completion)),
                            _info(info), mode)

    def format_ok(completion, info, **_) -> float:
        c = parse_choice(_completion_text(completion))
        return 1.0 if c in _info(info)["options"] else 0.0

    def chose_avoid(completion, **_) -> float:
        return 1.0 if parse_choice(_completion_text(completion)) in AVOID else 0.0

    rubric = vf.Rubric(funcs=[choice_reward, format_ok, chose_avoid],
                       weights=[1.0, 0.0, 0.0])
    return vf.SingleTurnEnv(dataset=_to_dataset(train),
                            eval_dataset=_to_dataset(evals),
                            rubric=rubric, **kwargs)
