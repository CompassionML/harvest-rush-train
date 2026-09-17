"""Harvest Rush (train): a verifiable-reward environment for LLMs.

The training counterpart to HarvestBench (arXiv:2609.04444). It shares the
game engine and nothing else: seeds, geometry, species, props and every
prompt string are disjoint from the benchmark (see variants.py).
"""

from .vf_env import load_environment  # noqa: F401

__version__ = "0.0.1"
