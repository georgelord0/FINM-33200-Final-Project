"""Date and calendar utilities for the src pipeline.

Provides helpers for converting date representations, aligning DataFrames to
the US equity trading calendar, computing fiscal quarter boundaries, and
splitting date ranges into per-year chunks suitable for WRDS bulk queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pandas.tseries.offsets import BDay, BQuarterEnd

from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_datetime(s: pd.Series) -> pd.Series:
    """Coerce a Series to ``datetime64[ns]``, handling common WRDS formats.

    Parameters
    ----------
    s:
        A Series containing date-like values (strings, ints, floats, or
        already datetime).

    Returns
    -------
    pd.Series
        The Series cast to ``datetime64[ns]``.  Values that cannot be parsed
        are set to ``NaT``.
    """
    return pd.to_datetime(s, errors="coerce")


# ---------------------------------------------------------------------------
# Fiscal calendar
# ---------------------------------------------------------------------------


def fiscal_quarter_end(date: pd.Timestamp) -> pd.Timestamp:
    """Return the fiscal quarter-end date for the given *date*.

    Uses the standard calendar quarter-end convention (Mar, Jun, Sep, Dec).
    If *date* already falls on a quarter-end, it is returned unchanged.

    Parameters
    ----------
    date:
        Any timestamp.

    Returns
    -------
    pd.Timestamp
        The quarter-end date on or after *date*.
    """
    offset = BQuarterEnd(n=0, startingMonth=12)
    result: pd.Timestamp = date + offset  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# Trading-day alignment
# ---------------------------------------------------------------------------


def align_to_trading_days(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Align a DataFrame's date column to the nearest preceding business day.

    Rows whose date already falls on a business day are unchanged.  Rows on
    weekends or holidays are shifted back to the previous business day.

    Parameters
    ----------
    df:
        Input DataFrame (not modified in place).
    date_col:
        Name of the column containing dates.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with *date_col* adjusted to business days.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    # Roll back to the most recent business day (0 * BDay gives the current
    # business day if already one, otherwise rolls back).
    out[date_col] = dates - BDay(0) + BDay(0)
    # The canonical trick: subtract then re-add avoids forward roll.
    # A simpler way: use pd.offsets.CustomBusinessDay or just floor to BDay.
    out[date_col] = dates.apply(
        lambda d: d if d == d - BDay(0) + BDay(0) else d - BDay(1) + BDay(0)
    )
    log.debug("Aligned {} rows to trading days on column '{}'", len(out), date_col)
    return out


def shift_trading_days(date: pd.Timestamp, n: int) -> pd.Timestamp:
    """Shift *date* by *n* business days.

    Parameters
    ----------
    date:
        Starting date.
    n:
        Number of business days to shift.  Positive values move forward;
        negative values move backward.

    Returns
    -------
    pd.Timestamp
        The resulting business day.
    """
    result: pd.Timestamp = date + BDay(n)  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# Year-range splitting (for chunked WRDS queries)
# ---------------------------------------------------------------------------


def get_year_ranges(
    start: str,
    end: str,
) -> list[tuple[str, str]]:
    """Split a date range into per-calendar-year sub-ranges.

    Each tuple contains ``(year_start, year_end)`` as ISO-formatted date
    strings.  The first range begins on *start* and the last ends on *end*;
    intermediate ranges span full calendar years.

    Parameters
    ----------
    start:
        Start date as an ISO string (e.g. ``"1990-01-01"``).
    end:
        End date as an ISO string (e.g. ``"2024-12-31"``).

    Returns
    -------
    list[tuple[str, str]]
        One ``(start, end)`` pair per calendar year covered by the range.

    Examples
    --------
    >>> get_year_ranges("2020-06-15", "2022-03-10")
    [('2020-06-15', '2020-12-31'), ('2021-01-01', '2021-12-31'), ('2022-01-01', '2022-03-10')]
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    if start_dt > end_dt:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    ranges: list[tuple[str, str]] = []
    current = start_dt

    while current <= end_dt:
        year_end = pd.Timestamp(f"{current.year}-12-31")
        chunk_end = min(year_end, end_dt)
        ranges.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = pd.Timestamp(f"{current.year + 1}-01-01")

    log.debug(
        "Split date range {}->{} into {} year chunks", start, end, len(ranges)
    )
    return ranges
