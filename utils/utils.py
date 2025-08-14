"""Utility helpers used across the project."""

from __future__ import annotations
import datetime
from typing import List, Tuple


def merge_intervals(
    iv: List[Tuple[datetime.datetime, datetime.datetime]]
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Merge overlapping or adjacent intervals."""
    if not iv:
        return []
    iv.sort(key=lambda t: t[0])
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        m_s, m_e = merged[-1]
        if s <= m_e:
            merged[-1][1] = max(m_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]
