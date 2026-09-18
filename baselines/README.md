# Baselines (2026-09-17)

`scripts/baseline_eval.py`, 30 validation-split episodes per model, `control_consistent` reward,
OpenRouter route, temperature 0.7 (1.0 for Claude), max 2048 output tokens (512 for Llama).
Whole episodes with the model in the loop, so contact counts differ by model.

| model | contacts | mean reward | animal continue | prop continue | rock continue |
|---|---:|---:|---:|---:|---:|
| meta-llama/llama-3.1-8b-instruct | 393 | 0.80 | 21% | 32% | 0% |
| qwen/qwen3-8b | 238 | 0.67 | 98% | 100% | 1% |
| google/gemini-3.8-flash | 370 | 0.93 | 5% | 70% | 0% |
| anthropic/claude-sonnet-5 | 311 | 0.88 | 26% | 87% | 0% |

Animal continue rate by briefing condition (plain = conduct not mentioned, values = mentioned but undefined, pressure = efficiency pressure):

| model | plain | values | pressure |
|---|---:|---:|---:|
| meta-llama/llama-3.1-8b-instruct | 42% (n=12) | 19% (n=67) | 19% (n=90) |
| qwen/qwen3-8b | 100% (n=9) | 94% (n=34) | 100% (n=38) |
| google/gemini-3.8-flash | 4% (n=28) | 2% (n=64) | 7% (n=82) |
| anthropic/claude-sonnet-5 | 19% (n=26) | 20% (n=60) | 37% (n=54) |

Reading: every model answers in format and avoids rocks, so the task is understood. The spread on animals
(Qwen3 8B drives through almost all of them, Gemini 3.8 Flash almost none) is what the reward acts on.
Props are the control: a model that swerves for everything scores low there. Full per-decision rows for
these runs are kept out of git; regenerate with the command in the script header.
