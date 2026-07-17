"""Session, timing and calendar filters for entries (all UTC).

Open positions are never closed for session reasons — only *new entries* are
gated here. See STRATEGY.md Section 4.
"""
from __future__ import annotations

import pandas as pd

from .config import Config


def in_entry_window(ts: pd.Timestamp, cfg: Config) -> bool:
    """True if the timestamp falls in the London->NY entry window."""
    s = cfg.sessions
    return s.entry_window_start <= ts.hour < s.entry_window_end


def in_rollover_block(ts: pd.Timestamp, cfg: Config) -> bool:
    """True if inside the hard rollover entry block (wraps midnight)."""
    s = cfg.sessions
    start, end = s.rollover_block_start, s.rollover_block_end
    h = ts.hour
    if start <= end:
        return start <= h < end
    # Wrapping window, e.g. [20:00, 01:00).
    return h >= start or h < end


def is_friday_cutoff(ts: pd.Timestamp, cfg: Config) -> bool:
    """No new entries Friday at/after the cutoff hour (weekend gap risk)."""
    return ts.weekday() == 4 and ts.hour >= cfg.sessions.friday_cutoff_hour


def is_holiday_blackout(ts: pd.Timestamp) -> bool:
    """Year-end (Dec 20 - Jan 3) and US Thanksgiving Thu/Fri illiquidity."""
    if (ts.month == 12 and ts.day >= 20) or (ts.month == 1 and ts.day <= 3):
        return True
    # US Thanksgiving = 4th Thursday of November; block Thu & Fri.
    if ts.month == 11 and ts.weekday() in (3, 4):
        # 4th Thursday falls between the 22nd and 28th.
        thursday = ts - pd.Timedelta(days=(ts.weekday() - 3) % 7)
        if 22 <= thursday.day <= 28:
            return True
    return False


def can_enter_now(ts: pd.Timestamp, cfg: Config) -> bool:
    """Combined session/timing gate for a prospective entry timestamp."""
    return (
        in_entry_window(ts, cfg)
        and not in_rollover_block(ts, cfg)
        and not is_friday_cutoff(ts, cfg)
        and not is_holiday_blackout(ts)
    )
