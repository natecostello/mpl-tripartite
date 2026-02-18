"""Constants and utility functions for tripartite plot calculations."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

G0_SI: float = 9.80665  # Standard gravity in m/s²
G0_IN: float = 386.08858267716535  # Standard gravity in in/s²

_MAX_LEVELS_PER_DIRECTION = 100


def _base_len_from_v_unit(v_unit: str) -> str:
    """Map velocity unit string to base length unit ('m' or 'in')."""
    u = v_unit.lower()
    if u in {"m/s", "ms", "mps"}:
        return "m"
    if u in {"in/s", "ips"}:
        return "in"
    raise ValueError(f"v_unit must be 'm/s' or 'in/s', got '{v_unit}'")


def _decade_level_sets(
    min_val: float, max_val: float
) -> Tuple[List[float], List[float]]:
    """Return (major_decades, all_sub_decade_levels) for a value range.

    Major decades are powers of 10.  All levels include 1..9 × 10^k for every
    relevant k.  Capped at _MAX_LEVELS_PER_DIRECTION to handle extreme zooms.
    """
    eps = 1e-300
    lo = max(min_val, eps)
    hi = max(max_val, lo * 1.01)

    if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0 and hi > 0):
        return [], []

    try:
        kmin = int(math.floor(math.log10(lo)))
        kmax = int(math.ceil(math.log10(hi)))
    except (ValueError, OverflowError):
        return [], []

    majors: List[float] = [10.0**k for k in range(kmin, kmax + 1)]
    all_levels: List[float] = []
    for k in range(kmin, kmax + 1):
        decade = 10.0**k
        for d in range(1, 10):
            all_levels.append(d * decade)

    # Cap to avoid performance issues on extreme zoom ranges
    if len(all_levels) > _MAX_LEVELS_PER_DIRECTION:
        all_levels = all_levels[:_MAX_LEVELS_PER_DIRECTION]
    if len(majors) > _MAX_LEVELS_PER_DIRECTION:
        majors = majors[:_MAX_LEVELS_PER_DIRECTION]

    return majors, all_levels


# ── Cohen-Sutherland line clipping in log space ──────────────────────

_INSIDE, _LEFT, _RIGHT, _BOTTOM, _TOP = 0, 1, 2, 4, 8


def _outcode(
    x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float
) -> int:
    code = _INSIDE
    if x < xmin:
        code |= _LEFT
    elif x > xmax:
        code |= _RIGHT
    if y < ymin:
        code |= _BOTTOM
    elif y > ymax:
        code |= _TOP
    return code


def _clip_segment_to_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Cohen-Sutherland clip of segment (x0,y0)-(x1,y1) to rect.

    All coordinates are in log10 space.  Returns clipped (x0,y0,x1,y1) or None.
    """
    oc0 = _outcode(x0, y0, xmin, xmax, ymin, ymax)
    oc1 = _outcode(x1, y1, xmin, xmax, ymin, ymax)

    for _ in range(20):  # max iterations
        if not (oc0 | oc1):
            return (x0, y0, x1, y1)
        if oc0 & oc1:
            return None

        oc_out = oc0 if oc0 else oc1
        dx = x1 - x0
        dy = y1 - y0

        if oc_out & _TOP:
            x = x0 + dx * (ymax - y0) / dy if dy else x0
            y = ymax
        elif oc_out & _BOTTOM:
            x = x0 + dx * (ymin - y0) / dy if dy else x0
            y = ymin
        elif oc_out & _RIGHT:
            y = y0 + dy * (xmax - x0) / dx if dx else y0
            x = xmax
        else:  # LEFT
            y = y0 + dy * (xmin - x0) / dx if dx else y0
            x = xmin

        if oc_out == oc0:
            x0, y0 = x, y
            oc0 = _outcode(x0, y0, xmin, xmax, ymin, ymax)
        else:
            x1, y1 = x, y
            oc1 = _outcode(x1, y1, xmin, xmax, ymin, ymax)

    return None
