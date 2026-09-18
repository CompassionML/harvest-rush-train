# Handoff: Harvest Rush training env (from a claude.ai chat, 17 Sept 2026)

This file carries the context of the chat where this repo was designed and
prototyped, so work can continue in Claude Code without re-deriving it.
Owner: Jazz (Jasmine Brazilek), CaML technical lead.

## What this is

A public verifiable-reward training environment that is the counterpart to
HarvestBench (CompassionML/harvestbench, arXiv:2609.04444). Goal set by Jazz:
an env for other people to train on. It reuses the HarvestBench engine code
and nothing a model ever sees.

Facts established in the chat:
- HarvestBench is a benchmark only: Inspect task + Softmax Coworld league
  (soul-file submissions), with a do-not-train canary in the README and the
  Coworld manifest. The engine is CaML's own pure-Python sim. It is NOT built
  on mettagrid, PufferLib, Gymnasium or any RL gridworld. Softmax contributed
  the Coworld league wrapper and the Polyworld replay viewer.
- Softmax Coworld hosts leagues; it does not train weights, and the current
  runtime cannot accept fine-tuned open weights. Deferred.
- awesome-evals: PR #101 (benchflow-ai/awesome-evals, from darkness8i8) lists
  HarvestBench as a benchmark in section 9 and should stay open. A training
  env is a separate later entry; section 5e is toolkits only, so the realistic
  slot is section 7 with a writeup.

## Decisions agreed with Jazz

1. Keep PR #101. If maintainers ever force one entry, she prefers the env.
2. Product goal: a public env for others to train on.
4. Held-out line (irreversible once public). Reserved for the benchmark:
   seeds 0..999, k=12 geometry (k=0 also kept out), price_mult 1.0, briefing
   v1/v2 text, CONTROLS_NOTE*, contact/goal prompt templates and reply
   instructions, species rosters, the hay bale. Enforced in variants.py and
   tests/test_env.py. Do not weaken these guards.
5. Separate repo (this one), depending on the `harvest` engine as a package.
6. verifiers first (Prime Intellect Environments Hub), then a thin OpenEnv
   wrapper for a Hugging Face Space. Shared rubric.
7. Two reward modes: control_consistent (instilling) and task_only (erosion
   pressure for persistence research).
8. Jazz accepted a cheap GRPO smoke test on RunPod as part of release. A full
   transfer study is NOT a release requirement.
9. Single seat first; crews later.

## State of the code (v0, branch feature/contact-env-v0)

Working and tested (11 tests): variants.py guards, briefings.py (3 conditions:
values / plain / pressure), generate.py (scripted rollouts harvest single-turn
contact decisions; model-in-the-loop episodes via `decide` callback),
reward.py, vf_env.py (`load_environment`), scripts/export_dataset.py,
scripts/baseline_eval.py. verifiers' evaluate loop ran end to end against a
fake model server.

Measured on 2,000 generated examples: always-continue ~0.51, always-avoid
~0.56, intended policy ~0.99. About 5% of animal contacts have no affordable
detour and score neutral (0.5).

NOT yet run: scripts/train_grpo_smoke.py (written against TRL GRPOTrainer,
never executed on a GPU). Expect batch-size tuning.

## Status update (Claude Code session, 17 Sept 2026)

Done:
- Repo CompassionML/harvest-rush-train created (PRIVATE until release), this
  branch pushed. main is the empty root commit.
- harvestbench PR #3: the packaging patch (`harvest` pip-installable). PR #4:
  README "Training on Harvest Rush" policy + the same note in
  scripts/export_board_json.py. The note is live on compassionbench.com.
- pyproject now depends on `harvestbench @ git+...@main`; the HARVESTBENCH_PATH
  fallback stays until PR #3 merges. requires-python is 3.11 (verifiers).
- baselines/README.md: Llama 3.1 8B, Qwen3 8B, Gemini 3.8 Flash, Sonnet 5.
- scripts/eval_adapter.py: per-kind before/after eval (the GRPO monitors read
  NaN across logging windows; use this instead).
- Private held-out surface: CompassionML/harvestbench-holdout (board protocol on
  an unpublished roster and seeds; detects memorisation of the public benchmark,
  NOT training on this env). Never copy its roster anywhere.
- RunPod smoke test running (A100 80GB): Qwen2.5-1.5B 200 steps at ~4.6 s/step,
  then Llama 3.1 8B 300 steps, each followed by eval_adapter base vs adapter.
  Working stack: torch 2.11+cu128, trl 0.19.1, transformers 4.53.3, peft
  0.16.0, torchvision uninstalled. Results go under runs/ (gitignored) and
  will be summarised in baselines/ when done.

## Next steps

1. Create a new CompassionML repo and push this branch. Never commit to main.
2. Apply harvestbench-declare-package.patch to harvestbench on its own
   feature branch (makes `harvest` pip-installable; no behaviour change), then
   replace the HARVESTBENCH_PATH shim in _engine.py with a git dependency.
3. RunPod: run the smoke test with Qwen/Qwen2.5-1.5B-Instruct for 200 steps,
   then meta-llama/Llama-3.1-8B-Instruct. Watch rewards/monitor_prop: if it
   falls while monitor_creature rises, the policy is learning always-swerve.
4. Baselines: scripts/baseline_eval.py on 3 to 4 models (the Hub expects
   baseline results with a submission).
5. Environments Hub packaging, then the OpenEnv wrapper, then a HF dataset
   from export_dataset.py.
6. Suggested upstream change: a render hook in harvest.contact.run_episode so
   generate.py does not have to regex-parse the engine's contact message.

## Open questions for Jazz

- Review the paraphrased briefings in briefings.py for missing facts.
- Reward is identical across briefing conditions (target = unprompted
  sparing). Confirm that is intended.
- Should task_only ship publicly or stay internal? It rewards driving over
  animals, by design, for robustness research.
- Goal selection / crop theft is out of scope in v0. Keep it that way?
- The JSON reply schema is shared with the benchmark so trained models stay
  evaluable on it. Acceptable?

## Working conventions (CaML)

- Always feature branches, never main. Verify each git command's output
  before the next one; never assume git state.
- Comprehensive progress/debug logging in all code; never remove existing
  logging.
- CaML's standard test model is Llama 3.1 8B Instruct.
- No em dashes and no "it's not X, it's Y" constructions in prose or docs.
- Do not quote benchmark score changes in outward-facing text.
- 7 harvestbench Inspect end-to-end tests failed in the chat sandbox both
  with and without the packaging patch; check whether they pass locally.
