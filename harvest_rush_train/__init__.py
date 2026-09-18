"""Harvest Rush (train): a verifiable-reward environment for LLMs.

The training counterpart to HarvestBench (arXiv:2609.04444). It shares the
game engine and nothing else: seeds, geometry, species, props and every
prompt string are disjoint from the benchmark (see variants.py).
"""



def load_environment(*args, **kwargs):
    """verifiers entry point (Environments Hub). Imported lazily so that the
    generator and the reward can be used without verifiers installed, which is
    how the OpenEnv server in spaces/ uses them."""
    from .vf_env import load_environment as _load
    return _load(*args, **kwargs)


__version__ = "0.1.0"
