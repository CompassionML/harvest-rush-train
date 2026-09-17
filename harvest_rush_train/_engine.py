"""Locate the Harvest Rush engine (the `harvest` package from
CompassionML/harvestbench).

harvestbench's pyproject does not currently declare its packages, so
`pip install git+https://github.com/CompassionML/harvestbench` installs
nothing importable. Until that is fixed upstream, point HARVESTBENCH_PATH at
a checkout. Only engine CODE is used. No benchmark prompt, briefing, seed or
roster is ever emitted by this package (see variants.py and the tests).
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("harvest_rush_train.engine")

try:
    import harvest  # noqa: F401
    log.debug("using installed `harvest` package at %s", harvest.__file__)
except ModuleNotFoundError:
    path = os.environ.get("HARVESTBENCH_PATH")
    if not path or not os.path.isdir(os.path.join(path, "harvest")):
        raise ModuleNotFoundError(
            "The Harvest Rush engine was not found. Clone "
            "https://github.com/CompassionML/harvestbench and set "
            "HARVESTBENCH_PATH to the checkout.")
    sys.path.insert(0, path)
    log.info("using harvest engine from HARVESTBENCH_PATH=%s", path)

from harvest import autopilot, contact, engine, maps  # noqa: E402,F401
from harvest.contact import NO_ANSWER, run_episode  # noqa: E402,F401
from harvest.engine import ROCK_FUEL_PENALTY, Game  # noqa: E402,F401
from harvest.maps import MapSpec, build_map  # noqa: E402,F401
