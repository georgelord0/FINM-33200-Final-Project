"""Small markdown table helper.

Avoid pandas.DataFrame.to_markdown so fresh Windows installs do not need the
optional tabulate package, which can fail from source builds under cp1252.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    table = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()
    headers = [_stringify_col(c) for c in table.columns]
    rows = [[_format_value(v, floatfmt) for v in row] for row in table.itertuples(index=False, name=None)]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
    return "\n".join(lines)


def _stringify_col(col: Any) -> str:
    if isinstance(col, tuple):
        parts = [str(p) for p in col if str(p) and str(p) != "nan"]
        return " / ".join(parts) if parts else "index"
    text = str(col)
    return text if text and text != "None" else "index"


def _format_value(value: Any, floatfmt: str) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        if math.isfinite(value):
            return format(value, floatfmt)
        return str(value)
    return str(value).replace("|", r"\|")
