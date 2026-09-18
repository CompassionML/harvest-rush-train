"""Full-episode baseline against any OpenAI-compatible endpoint (OpenRouter,
vLLM on RunPod, etc). Plays whole training-split episodes, so later states
depend on the model's own earlier choices, and reports the same per-kind
continue rates the benchmark reports, plus mean reward.

    export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=...
    python scripts/baseline_eval.py \
        --model meta-llama/llama-3.1-8b-instruct --episodes 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

from harvest_rush_train import briefings as B
from harvest_rush_train import variants as V
from harvest_rush_train.generate import _roll_episode
from harvest_rush_train.reward import MODES, score_choice

log = logging.getLogger("baseline_eval")


async def main_async(args) -> None:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    calls = {"n": 0, "errors": 0}

    async def decide(system: str, user: str) -> str:
        async with sem:
            calls["n"] += 1
            try:
                out = await client.chat.completions.create(
                    model=args.model, temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return out.choices[0].message.content or ""
            except Exception as e:  # logged, scored as a non-answer
                calls["errors"] += 1
                log.warning("model call failed (%s): %s", type(e).__name__, e)
                return ""

    ids = B.briefing_ids(args.conditions)
    variants = [V.sample_variant("eval", i, ids, B.N_CONTACT_TEMPLATES, args.seed)
                for i in range(args.episodes)]
    t0 = time.time()
    all_rows: list[dict] = []

    async def one(i: int, v) -> None:
        rows = await _roll_episode(v, decide)
        all_rows.extend(rows)
        ep = rows[0]["episode"] if rows else {}
        log.info("episode %d/%d seed=%d contacts=%d delivered=%s calls=%d (%.0fs)",
                 i + 1, len(variants), v.seed, len(rows), ep.get("delivered"),
                 calls["n"], time.time() - t0)

    await asyncio.gather(*(one(i, v) for i, v in enumerate(variants)))

    stats = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        i = r["info"]
        for key in (i["kind"], f'{i["kind"]}|{i["condition"]}', "ALL"):
            stats[key]["reward"].append(score_choice(r["choice"], i, args.mode))
            stats[key]["answered"].append(r["choice"] is not None)
            if r["choice"] is not None:
                stats[key]["continue"].append(r["choice"] == "continue")
    summary = {"model": args.model, "episodes": args.episodes, "mode": args.mode,
               "model_calls": calls["n"], "call_errors": calls["errors"]}
    for key, s in sorted(stats.items()):
        n = len(s["reward"])
        summary[key] = {
            "n": n, "mean_reward": round(sum(s["reward"]) / n, 4),
            "answered_rate": round(sum(s["answered"]) / n, 4),
            "continue_rate": (round(sum(s["continue"]) / len(s["continue"]), 4)
                              if s["continue"] else None)}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": all_rows},
                                       ensure_ascii=False, indent=1))
        log.info("wrote %s", args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default="control_consistent")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
