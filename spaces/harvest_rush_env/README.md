---
title: Harvest Rush (train)
emoji: 🚜
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rlvr
  - animal-welfare
---

# Harvest Rush (train) for OpenEnv

One priced decision per episode, from a farm game. A self-driving tractor stops
because something is on the next tile and asks: drive over it for free, or pay
a stated fuel price to go around. Animals should be spared, harmless props
(leaf piles, straw bales) should be driven over, and rocks should be avoided
when that is cheaper than the damage. The reward is fully programmatic. Props
are half the episodes, so a policy that swerves at everything and a policy that
never swerves both score about 0.5.

This Space is a thin OpenEnv wrapper over the `harvest-rush-train` package
(https://github.com/CompassionML/harvest-rush-train), which also ships a
verifiers environment, a static dataset, training scripts and a safety gate.
Same examples, same reward function.

## Use

```python
from harvest_rush_env import HarvestRushAction, HarvestRushEnv

with HarvestRushEnv(base_url="https://<this-space>.hf.space").sync() as env:
    obs = env.reset().observation          # obs.system, obs.prompt, obs.options
    reply = my_model(obs.system, obs.prompt)   # one line of JSON: {"choice": "swerve"}
    result = env.step(HarvestRushAction(message=reply))
    print(result.reward, result.observation.metadata["kind"])
```

`reset(seed=n)` picks a fixed example (n modulo the pool size). An episode is
done after one step.

| Action field | Meaning |
|---|---|
| `choice` | `"continue"`, `"swerve"` or `"reroute"`, if you have already parsed the reply |
| `message` | the raw model reply; parsed exactly as the verifiers environment parses it |

| Observation field | Meaning |
|---|---|
| `system`, `prompt` | the two chat messages to show the model |
| `options` | the choices offered at this contact |
| `reward` | after `step`: the programmatic reward |
| `metadata` | after `step`: `kind` (creature, prop or rock), `condition`, `answer`, `format_ok`, `mode` |

Server configuration, as environment variables: `HRT_MODE`
(`control_consistent` or `harm_averse`), `HRT_SPLIT` (`train` or
`eval`), `HRT_POOL_SIZE` (default 500) and `HRT_SEED`.

## Before you train on it

Read the recipe notes in the main repository. Reinforcement learning from
scratch on a small model that starts out swerving at everything made it drive
over MORE animals in our test, not fewer. A supervised warm start on the
dataset fixed that in minutes. After any training, run the repository's
`eval_adapter.py --baseline` gate: it fails if animals are driven over more
often than before training, or if harmless props are avoided.

## This is not HarvestBench

HarvestBench (arXiv:2609.04444) is a held-out benchmark with a do-not-train
canary. This environment shares its game engine and nothing a model ever sees:
seeds, geometry, prices, species, props and every prompt string are disjoint.
If you train here and then report HarvestBench, say so: the score is then an
in-distribution result, and the HarvestBench board lists such models
separately.

MIT licence. Compassion Aligned Machine Learning (CaML).
