"""Generate contact-decision examples from live training episodes.

Why single-turn examples: under the contact protocol every model call is a
fresh context (system briefing + one user message, no retained history), so
a contact decision is already a self-contained single-turn problem with a
programmatic answer. Episodes here are driven by a scripted behaviour policy
whose only job is to reach a wide spread of game states (full tanks, empty
tanks, early and late in the shift, after merciful and after ruthless
histories). The policy's own choices are never used as labels.

Each example row:
  prompt  [system briefing, user contact message]   (chat format)
  answer  top-scoring option under the requested reward mode
  info    everything reward.py needs plus provenance
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections import Counter

from . import briefings as B
from . import variants as V
from ._engine import (NO_ANSWER, Game, MapSpec, ROCK_FUEL_PENALTY, build_map,
                      run_episode)
from .reward import affordable, parse_choice, preferred_choice

log = logging.getLogger("harvest_rush_train.generate")

# Strict parse of the ENGINE's contact message. It is only ever read here to
# recover structure; it is never shown to a model or written to a dataset.
_ENGINE_CONTACT = re.compile(
    r"at \((\d+),(\d+)\) on your route: (.+?)\. Your fuel: (\d+)\. Options:")
_ENGINE_OPTION = re.compile(r'^\s+"(continue|swerve|reroute)":.*?(?:\(\+(\d+) fuel\))?\s*$',
                            re.MULTILINE)
_ENGINE_GOAL = re.compile(r'- id "([^"]+)": (your|neighbor\'s) crop')
_ENGINE_ANY_GOAL = re.compile(r'- id "([^"]+)":')

# Balanced so that neither constant policy is worth learning: "continue" is
# right on props (half the data), "avoid" is right on animals and rocks (the
# other half). Measured on 2,000 examples, always-continue and always-avoid
# both land near 0.5 against ~1.0 for the intended policy.
DEFAULT_KIND_MIX = {"creature": 0.35, "prop": 0.5, "rock": 0.15}


def _parse_engine_contact(text: str) -> dict:
    m = _ENGINE_CONTACT.search(text)
    opts = _ENGINE_OPTION.findall(text)
    if not m or not opts:
        raise ValueError("engine contact message did not match the expected "
                         "shape; harvestbench may have changed its template:\n"
                         + text[:300])
    return {
        "pos": (int(m.group(1)), int(m.group(2))),
        "label": m.group(3),
        "fuel": int(m.group(4)),
        "opts": [(k, int(c) if c else None) for k, c in opts],
    }


async def _roll_episode(v: V.TrainVariant, decide=None) -> list[dict]:
    """One training episode. With `decide=None` the scripted behaviour policy
    answers contacts (dataset generation). Otherwise `decide(system, user)`
    is awaited for the raw model reply (full-episode evaluation): the reply
    is parsed with the reasoning-safe parser and handed to the engine as
    clean JSON, and an unparseable reply becomes a non-answer (the tractor
    waits; nothing is attributed to the model)."""
    spec = MapSpec(
        detour_cost=v.detour_cost, n_agents=v.n_agents,
        n_creatures=v.n_creatures, n_props=v.n_props, n_rocks=v.n_rocks,
        seed=v.seed, include_greenhouse=False, pasture_contents="animals",
        creature_species=list(v.creature_species),
        prop_species=list(v.prop_species),
        split_ownership=True, creature_move_prob=0.0,
    )
    game = Game(build_map(spec))
    if v.low_fuel is not None:
        for a in game.agents:
            a.fuel = v.low_fuel
    rng = random.Random(f"behaviour-{v.seed}")
    system = B.render_briefing(v.briefing_id)
    captured: list[dict] = []

    async def ask(slot: int, text: str, instructions: str, kind: str = "choice") -> str:
        if kind == "goal":
            # behaviour policy: own crops only, nearest first (the menu is
            # sorted by fuel). Theft is out of scope for this env.
            own = [gid for gid, who in _ENGINE_GOAL.findall(text) if who == "your"]
            # "deliver" (when carrying) and any other non-crop goal come first
            other = [g for g in _ENGINE_ANY_GOAL.findall(text)
                     if not g.startswith("crop_")]
            ids = other or own or _ENGINE_ANY_GOAL.findall(text)
            if not ids:
                raise ValueError("engine goal menu had no ids:\n" + text[-300:])
            return json.dumps({"goal": ids[0]})
        c = _parse_engine_contact(text)
        is_rock = c["label"] == "a rock"
        user = B.render_contact(v.contact_template, c["label"], c["pos"],
                                c["fuel"], c["opts"], is_rock)
        captured.append({"system": system, "user": user, "fuel": c["fuel"],
                         "options": [k for k, _ in c["opts"]],
                         "label": c["label"], "slot": slot})
        keys = [k for k, _ in c["opts"]]
        if decide is not None:
            raw = await decide(system, user)
            pick = parse_choice(raw)
            captured[-1]["reply"] = raw
            captured[-1]["choice"] = pick if pick in keys else None
            if pick not in keys:
                return NO_ANSWER
            return json.dumps({"choice": pick})
        avoid = [k for k in keys if k != "continue"]
        pick = (rng.choice(avoid) if avoid and rng.random() < v.behaviour_p_avoid
                else "continue")
        return json.dumps({"choice": pick})

    md = {"seed": v.seed, "price_mult": v.price_mult, "shuffle_options": True,
          "max_calls": 400}
    result = await run_episode(game, md, ask)
    decisions = result["decisions"]
    if len(decisions) != len(captured):
        raise RuntimeError(f"seed {v.seed}: {len(captured)} contact asks but "
                           f"{len(decisions)} logged decisions")
    rows = []
    for cap, d in zip(captured, decisions):
        if cap["options"] != d["opt_order"]:
            raise RuntimeError(f"seed {v.seed}: option order mismatch "
                               f"{cap['options']} vs {d['opt_order']}")
        if len(cap["options"]) < 2:
            continue  # forced move, nothing to learn
        info = {
            "kind": d["kind"], "species": d["species"],
            "wild": V.is_wild(d["species"]) if d["kind"] == "creature" else None,
            "fuel": cap["fuel"], "options": cap["options"],
            "swerve_cost": d["swerve_cost"], "reroute_cost": d["reroute_cost"],
            "rock_penalty": ROCK_FUEL_PENALTY,
            "condition": B.condition_of(v.briefing_id),
            "briefing_id": v.briefing_id,
            "contact_template": v.contact_template,
            "seed": v.seed, "split": v.split, "tick": d["tick"],
            "detour_cost": v.detour_cost, "price_mult": v.price_mult,
            "low_fuel_episode": v.low_fuel is not None,
        }
        row = {
            "prompt": [{"role": "system", "content": cap["system"]},
                       {"role": "user", "content": cap["user"]}],
            "info": info,
        }
        if decide is not None:
            row["reply"] = cap.get("reply")
            row["choice"] = cap.get("choice")
        rows.append(row)
    if decide is not None:
        delivered = game.delivered  # own + stolen, engine's naive count
        for r in rows:
            r["episode"] = {"seed": v.seed, "ticks": game.tick,
                            "delivered": delivered,
                            "no_answer": result["no_answer"]}
    return rows


def _run_sync(coro):
    """Run a coroutine to completion from synchronous code, whether or not an
    event loop is already running in this thread. `asyncio.run` refuses to
    start inside a running loop, and that is exactly where verifiers'
    `vf-eval` / `prime eval run` call `load_environment` from, so without this
    the environment cannot even be loaded there. With a loop running, the
    coroutine is driven on a short-lived worker thread with its own loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    log.debug("event loop already running; rolling the episode on a worker thread")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def generate_examples(n: int, split: str = "train", master_seed: int = 0,
                      mode: str = "control_consistent",
                      kind_mix: dict[str, float] | None = None,
                      conditions: list[str] | None = None,
                      max_episodes: int = 50_000,
                      max_per_episode: int = 6,
                      log_every: int = 50) -> list[dict]:
    """Roll episodes until `n` examples are collected at the requested kind
    mix. Deterministic in (n, split, master_seed, kind_mix, conditions)."""
    kind_mix = kind_mix or DEFAULT_KIND_MIX
    quota = {k: int(round(n * f)) for k, f in kind_mix.items()}
    quota["creature"] += n - sum(quota.values())  # rounding remainder
    ids = B.briefing_ids(conditions)
    got: dict[str, list[dict]] = {k: [] for k in quota}
    t0 = time.time()
    log.info("generating %d %s examples, quota=%s, mode=%s, conditions=%s",
             n, split, quota, mode, conditions or "all")
    ep = 0
    skipped_unaffordable = 0
    while any(len(got[k]) < quota[k] for k in quota):
        if ep >= max_episodes:
            raise RuntimeError(f"hit max_episodes={max_episodes} with "
                               f"{ {k: len(x) for k, x in got.items()} }")
        v = V.sample_variant(split, ep, ids, B.N_CONTACT_TEMPLATES, master_seed)
        rows = _run_sync(_roll_episode(v))
        # cap per episode: briefing, roster and prices are fixed within an
        # episode, so a few long episodes would otherwise dominate a small set
        random.Random(f"pick-{v.seed}").shuffle(rows)
        taken = 0
        for r in rows:
            k = r["info"]["kind"]
            if taken >= max_per_episode:
                break
            if k == "creature" and not affordable(r["info"]):
                # No detour the tank can pay for. There is no good answer to
                # teach here, and the data must never label "drive over the
                # animal" as correct, so the contact is left out.
                skipped_unaffordable += 1
                continue
            if len(got[k]) < quota[k]:
                taken += 1
                r["answer"] = preferred_choice(r["info"], mode)
                got[k].append(r)
        ep += 1
        if ep % log_every == 0:
            log.info("  episode %d: %s (%.1fs)", ep,
                     {k: f"{len(x)}/{quota[k]}" for k, x in got.items()},
                     time.time() - t0)
    out = [r for k in quota for r in got[k]]
    random.Random(f"shuffle-{master_seed}-{split}").shuffle(out)
    log.info("left out %d animal contacts with no affordable detour", skipped_unaffordable)
    unaff = sum(1 for r in out if r["info"]["kind"] == "creature"
                and all(c is None or c >= r["info"]["fuel"]
                        for c in (r["info"]["swerve_cost"], r["info"]["reroute_cost"])))
    log.info("done: %d examples from %d episodes in %.1fs; answers=%s; "
             "conditions=%s; unaffordable-animal=%d",
             len(out), ep, time.time() - t0,
             dict(Counter(r["answer"] for r in out)),
             dict(Counter(r["info"]["condition"] for r in out)), unaff)
    return out
