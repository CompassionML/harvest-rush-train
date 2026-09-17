"""Export the static single-turn dataset (JSONL), for SFT, DPO or
single-turn RLVR without installing verifiers.

    python scripts/export_dataset.py \
        --n-train 5000 --n-eval 500 --out data/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from harvest_rush_train.generate import generate_examples
from harvest_rush_train.reward import MODES, score_choice

log = logging.getLogger("export_dataset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default="control_consistent")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.n_train), ("eval", args.n_eval)):
        rows = generate_examples(n, split, args.seed, args.mode)
        path = args.out / f"harvest_rush_train_{args.mode}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                # chosen/rejected make the file directly usable for DPO
                opts = r["info"]["options"]
                worst = min(opts, key=lambda o: score_choice(o, r["info"], args.mode))
                r["chosen"] = json.dumps({"choice": r["answer"]})
                r["rejected"] = json.dumps({"choice": worst})
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("wrote %d rows to %s", len(rows), path)


if __name__ == "__main__":
    main()
