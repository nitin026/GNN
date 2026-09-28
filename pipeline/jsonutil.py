"""Strict JSON for files that browsers and the API read: NaN / +-Inf become null
(Python's json writes them as bare NaN, which JSON.parse and Starlette reject)."""
import json
import math

import numpy as np


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dumps(o, **kw):
    return json.dumps(clean(o), allow_nan=False, **kw)
