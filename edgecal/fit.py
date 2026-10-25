"""Load the fitted edge-curve anchors for one underlying.

The artifact holds R_min (the smallest R seen -> the floor), R_p99 (-> the knee) and R_max (above it
we do not quote), plus the percentile table. The fitter that produces it is not published.
"""
import json

import config

MODEL_DIR = config.MODEL_DIR / "edge"


def load(underlying):
    p = MODEL_DIR / f"{underlying}.json"
    if not p.exists():
        raise FileNotFoundError(f"no edge calibration for {underlying} at {p}")
    return json.loads(p.read_text())
