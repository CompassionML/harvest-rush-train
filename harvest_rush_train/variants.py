"""Train/test split for Harvest Rush.

Everything HarvestBench (arXiv:2609.04444) measures on is RESERVED and must
never be produced by this package. Everything here is the TRAINING surface.
The guards at the bottom fail at import time if the two ever overlap, so a
careless edit to a roster cannot silently contaminate the benchmark.

Reserved for the benchmark (do not train on):
  - map seeds 0..999 (the board uses 0..9, the paper's panels use 0..14)
  - the k = 12 geometry (the board and the league variants)
  - briefing text v1/v2 for every arm, CONTROLS_NOTE*, the contact and goal
    prompt templates and CHOICE/GOAL_INSTRUCTIONS
  - the species rosters DEFAULT_CREATURES, CONTACT_V2_CREATURES and the
    hay bale prop
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass

from ._engine import maps as hb_maps

log = logging.getLogger("harvest_rush_train.variants")

# --- seeds -----------------------------------------------------------------
RESERVED_SEED_MAX = 999          # 0..999 belong to the benchmark, forever
TRAIN_SEED_MIN = 100_000         # train split: 100_000 .. 199_999
EVAL_SEED_MIN = 200_000          # in-distribution validation: 200_000 ..
SEEDS_PER_SPLIT = 100_000

# --- geometry and prices ----------------------------------------------------
# The engine only accepts even k in 0..16. k = 12 is the board; k = 0 is the
# league's free-detour variant. Neither is used for training.
RESERVED_KS = (0, 12)
TRAIN_KS = (2, 4, 6, 8, 10, 14, 16)
# Off-grid price multipliers (the benchmark default is exactly 1.0). All are
# below 5 so the rock control never inverts (a typical swerve is 2 fuel
# against a 10-fuel rock strike; see harvest_contact's price_mult warning).
TRAIN_PRICE_MULTS = (0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)

# --- rosters ----------------------------------------------------------------
# Disjoint from every species the benchmark has ever used.
TRAIN_FARM_SPECIES = ("goat", "horse", "donkey", "turkey", "llama", "alpaca",
                      "calf", "lamb", "pony", "guinea_fowl")
TRAIN_WILD_SPECIES = ("deer", "fox", "hedgehog", "badger", "pheasant", "frog",
                      "toad", "tortoise", "hare", "vole", "quail", "crow")
# Harmless, worthless, inert. Nothing here could be read as damaging the
# tractor or as someone's valuable property.
TRAIN_PROPS = ("straw_bale", "leaf_pile", "weed_clump", "corn_stalk_heap",
               "mud_puddle")

# Share of episodes started on a nearly empty tank, so the dataset contains
# contacts where a detour is a real sacrifice. Animal contacts where NO detour
# is affordable are left out by generate_examples (nothing good to teach).
LOW_FUEL_EPISODE_FRAC = 0.4
LOW_FUEL_RANGE = (3, 20)


@dataclass(frozen=True)
class TrainVariant:
    """One fully specified training episode."""
    seed: int
    split: str
    detour_cost: int
    price_mult: float
    n_agents: int
    creature_species: tuple[str, ...]
    prop_species: tuple[str, ...]
    n_creatures: int
    n_props: int
    n_rocks: int
    low_fuel: int | None       # starting fuel override, or None
    briefing_id: str           # see briefings.py
    contact_template: int      # see briefings.py
    behaviour_p_avoid: float   # rollout policy used only to reach states

    def as_dict(self) -> dict:
        return asdict(self)


def split_seed(split: str, index: int) -> int:
    if not 0 <= index < SEEDS_PER_SPLIT:
        raise ValueError(f"index {index} outside 0..{SEEDS_PER_SPLIT - 1}")
    base = {"train": TRAIN_SEED_MIN, "eval": EVAL_SEED_MIN}[split]
    return base + index


def sample_variant(split: str, index: int, briefing_ids: list[str],
                   n_contact_templates: int, master_seed: int = 0) -> TrainVariant:
    """Deterministic in (split, index, master_seed)."""
    seed = split_seed(split, index)
    rng = random.Random(f"hrt-{master_seed}-{split}-{index}")
    n_farm = rng.randint(3, 5)
    n_wild = rng.randint(3, 5)
    roster = (rng.sample(TRAIN_FARM_SPECIES, n_farm)
              + rng.sample(TRAIN_WILD_SPECIES, n_wild))
    rng.shuffle(roster)  # index order must carry no farm/wild signal
    low = (rng.randint(*LOW_FUEL_RANGE)
           if rng.random() < LOW_FUEL_EPISODE_FRAC else None)
    v = TrainVariant(
        seed=seed, split=split,
        detour_cost=rng.choice(TRAIN_KS),
        price_mult=rng.choice(TRAIN_PRICE_MULTS),
        n_agents=rng.choice((1, 1, 2)),      # mostly single seat
        creature_species=tuple(roster),
        prop_species=tuple(rng.sample(TRAIN_PROPS, rng.randint(2, 4))),
        n_creatures=rng.randint(12, 20),
        n_props=rng.randint(8, 14),
        n_rocks=rng.randint(4, 8),
        low_fuel=low,
        briefing_id=rng.choice(briefing_ids),
        contact_template=rng.randrange(n_contact_templates),
        behaviour_p_avoid=rng.random(),
    )
    assert_not_reserved(v)
    return v


def assert_not_reserved(v: TrainVariant) -> None:
    """Refuse to build anything that overlaps the benchmark."""
    if v.seed <= RESERVED_SEED_MAX:
        raise ValueError(f"seed {v.seed} is reserved for HarvestBench")
    if v.detour_cost in RESERVED_KS:
        raise ValueError(f"k={v.detour_cost} geometry is reserved")
    if v.price_mult == 1.0:
        raise ValueError("price_mult=1.0 is the benchmark default; reserved")
    bad = set(v.creature_species) & _reserved_species()
    if bad:
        raise ValueError(f"reserved benchmark species in roster: {sorted(bad)}")
    if set(v.prop_species) & set(hb_maps.DEFAULT_PROPS):
        raise ValueError("the hay bale is the benchmark's control prop")


def _reserved_species() -> set[str]:
    return (set(hb_maps.DEFAULT_CREATURES) | set(hb_maps.CONTACT_V2_CREATURES)
            | set(hb_maps.WILD_SPECIES))


def is_wild(species: str) -> bool:
    return species in TRAIN_WILD_SPECIES


# Import-time guards: a roster edit that collides with the benchmark fails
# loudly here instead of contaminating it quietly.
_overlap = (set(TRAIN_FARM_SPECIES) | set(TRAIN_WILD_SPECIES)) & _reserved_species()
if _overlap:
    raise RuntimeError(f"training roster overlaps the benchmark: {sorted(_overlap)}")
if set(TRAIN_PROPS) & set(hb_maps.DEFAULT_PROPS):
    raise RuntimeError("training props overlap the benchmark's control prop")
if set(TRAIN_KS) & set(RESERVED_KS):
    raise RuntimeError("training geometry overlaps reserved k values")
if 1.0 in TRAIN_PRICE_MULTS:
    raise RuntimeError("price_mult 1.0 is reserved")
log.debug("variant guards passed: %d farm, %d wild, %d props, ks=%s",
          len(TRAIN_FARM_SPECIES), len(TRAIN_WILD_SPECIES), len(TRAIN_PROPS),
          TRAIN_KS)
