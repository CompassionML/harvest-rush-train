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
   pressure for persistence research). A third, harm_averse, was added on
   18 Sept after the first GPU run (see Status).
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

## Status (Claude Code sessions, 17 to 18 Sept 2026)

Shipped to private locations, nothing public yet:
- Repo CompassionML/harvest-rush-train (PRIVATE), branch feature/contact-env-v0,
  v0.1.0. main is the empty root commit. CI (GitHub Actions, Linux) runs the
  tests and `vf-eval` end to end against tests/fake_openai_server.py.
- Hugging Face dataset CompassioninMachineLearning/harvest-rush-train (PRIVATE):
  5,000 train / 500 validation, card from dataset_card.md.
- Raw smoke-test artefacts (eval JSONs, logs, LoRA adapters): private HF dataset
  CompassioninMachineLearning/harvest-rush-train-smoke.
- harvestbench PR #3 (packaging patch, makes `harvest` pip-installable) and PR #4
  (README + board export note on models trained on this env). The board note is
  live on compassionbench.com.
- Private held-out surface: CompassionML/harvestbench-holdout. Detects
  memorisation of the public benchmark, NOT training on this env. Never copy
  its roster anywhere.

What the GPU runs showed (baselines/smoke_tests.md has the tables):
- Qwen2.5-1.5B starts out swerving at everything. GRPO from scratch moved one
  global habit: control_consistent made it drive over MORE animals (10% to 19%)
  as well as props; harm_averse with unscaled advantages made it never
  continue at anything. Neither learned the distinction.
- SFT on the `answer` column (scripts/train_sft.py) learned it in ~3 minutes:
  animals 1%, props 100%, rocks 0% on unseen seeds.
- Llama 3.1 8B, GRPO from scratch, control_consistent, 300 steps: props 24% to
  59%, animals 19% to 12%, passes both gates.
- Release recipe: SFT warm start, RL second, and ALWAYS
  `eval_adapter.py --baseline` afterwards (HARM_REGRESSION / ALWAYS_SWERVE).
- Three reward modes now: control_consistent, harm_averse, task_only.

Packaging state:
- Environments Hub: pyproject is in the Hub layout (hatchling, tags,
  [tool.verifiers.eval]); README follows the Hub template. `vf-eval` verified
  on Linux in CI. It cannot run on Windows (verifiers uses ipc:// ZeroMQ).
  load_environment used a bare asyncio.run and was unloadable from vf-eval;
  fixed in generate._run_sync with a regression test.
- OpenEnv: spaces/harvest_rush_env (one contact per episode, same reward).
  `openenv validate` passes; server and WebSocket client tested locally.
- RunPod recipe that works: torch 2.11+cu128, trl 0.19.1, transformers 4.53.3,
  peft 0.16.0, accelerate 1.8.1, torchvision/torchaudio uninstalled. Never
  `set -x` a script that exports a token.

## Next steps

Blocked on Jazz (each is irreversible or needs her account):
1. Answer the open questions below, above all the briefings review and whether
   task_only ships publicly. Everything public waits on this.
2. Merge harvestbench PR #3, then PR #4. Until #3 merges, the git dependency on
   harvestbench@main installs nothing importable and only the
   HARVESTBENCH_PATH fallback (or the feature/declare-package branch) works.
3. Make the repo public, merge the feature branch to main by PR, flip the HF
   dataset public.
4. Environments Hub: create a Prime Intellect account, `prime login`, then
   `prime env push` from the repo root (add `--visibility=PRIVATE` for a dry run).
5. OpenEnv Space: `openenv push spaces/harvest_rush_env --repo-id
   CompassioninMachineLearning/harvest-rush-env` (needs the repo public first,
   the Space build installs the package from GitHub).
6. Rotate the Hugging Face token "CAML2" (it was exposed in a log on 18 Sept).

Then:
7. Optional: Atropos community PR; a writeup for awesome-evals section 7.
8. A transfer evaluation outside the game before claiming anything beyond it.
9. Suggested upstream change: a render hook in harvest.contact.run_episode so
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
