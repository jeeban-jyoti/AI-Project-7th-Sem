"""Loading a policy for the live pipeline."""
from __future__ import annotations

import os
import sys

from tourism.m4_rl.agents import FIXED, Policy, RulePolicy, load_agent


def load_policy(name: str, models_dir: str) -> Policy:
    if name in FIXED:
        return FIXED[name]()
    path = os.path.join(models_dir, "rl_%s.json" % name)
    if os.path.exists(path):
        return load_agent(path)
    sys.stderr.write("[warn] no trained %s policy at %s (run ./run.sh train); using the simple rule instead\n"
                     % (name, path))
    return RulePolicy()
