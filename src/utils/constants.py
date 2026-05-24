"""Project-wide constants for the src pipeline.

Centralizes filtering criteria, date ranges, column name mappings,
and data type specifications used across CRSP, Compustat, and derived datasets.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Universe filters
# ---------------------------------------------------------------------------

COMMON_STOCK_SHRCDS: tuple[int, int] = (10, 11)
"""CRSP share codes for common stocks."""

MAJOR_EXCHANGES: tuple[int, int, int] = (1, 2, 3)
"""CRSP exchange codes: 1 = NYSE, 2 = AMEX, 3 = NASDAQ."""

MIN_PRICE: float = 1.0
"""Minimum share price filter (excludes penny stocks)."""

MIN_MARKET_CAP: float = 1e7
"""Minimum market capitalisation filter (USD 10 million)."""

# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Approximate number of trading days in a calendar year."""

TRADING_DAYS_PER_MONTH: int = 21
"""Approximate number of trading days in a calendar month."""

TRADING_DAYS_PER_QUARTER: int = 63
"""Approximate number of trading days in a calendar quarter."""

START_DATE: str = "1990-01-01"
"""Default start of the sample period."""

END_DATE: str = "2024-12-31"
"""Default end of the sample period."""

# ---------------------------------------------------------------------------
# Feature column names
# ---------------------------------------------------------------------------

# Identifiers
COL_PERMNO: str = "permno"
COL_GVKEY: str = "gvkey"
COL_DATE: str = "date"
COL_TICKER: str = "ticker"

# CRSP return / price columns
COL_RET: str = "ret"
COL_RETX: str = "retx"
COL_PRC: str = "prc"
COL_SHROUT: str = "shrout"
COL_VOL: str = "vol"
COL_BID: str = "bid"
COL_ASK: str = "ask"
COL_SPREAD: str = "spread"
COL_MARKET_CAP: str = "market_cap"

# Compustat fundamentals
COL_AT: str = "at"  # total assets
COL_LT: str = "lt"  # total liabilities
COL_SEQ: str = "seq"  # stockholders' equity
COL_CEQQ: str = "ceqq"  # common equity (quarterly)
COL_REVTQ: str = "revtq"  # revenue (quarterly)
COL_NIQ: str = "niq"  # net income (quarterly)
COL_IBQ: str = "ibq"  # income before extra items (quarterly)
COL_OIADPQ: str = "oiadpq"  # operating income (quarterly)
COL_DATADATE: str = "datadate"
COL_RDQ: str = "rdq"  # report date of quarterly earnings
COL_FYEARQ: str = "fyearq"
COL_FQTR: str = "fqtr"

# Derived features
COL_MOM_21: str = "mom_21"
COL_MOM_63: str = "mom_63"
COL_MOM_126: str = "mom_126"
COL_MOM_252: str = "mom_252"
COL_VOL_21: str = "vol_21"
COL_VOL_63: str = "vol_63"
COL_ILLIQ_21: str = "illiq_21"
COL_ILLIQ_63: str = "illiq_63"
COL_LOG_MARKET_CAP: str = "log_market_cap"
COL_BOOK_TO_MARKET: str = "book_to_market"
COL_ROE: str = "roe"
COL_ASSET_GROWTH: str = "asset_growth"
COL_GROSS_PROFITABILITY: str = "gross_profitability"

# Factor / benchmark columns
COL_MKT_RF: str = "mktrf"
COL_SMB: str = "smb"
COL_HML: str = "hml"
COL_RF: str = "rf"
COL_UMD: str = "umd"

# ---------------------------------------------------------------------------
# Dtype mappings — used when reading raw WRDS data to enforce correct types
# ---------------------------------------------------------------------------

CRSP_DTYPES: dict[str, str] = {
    "permno": "int64",
    "date": "datetime64[ns]",
    "ret": "float64",
    "retx": "float64",
    "prc": "float64",
    "shrout": "float64",
    "vol": "float64",
    "bid": "float64",
    "ask": "float64",
    "shrcd": "int16",
    "exchcd": "int16",
}

COMPUSTAT_DTYPES: dict[str, str] = {
    "gvkey": "str",
    "datadate": "datetime64[ns]",
    "rdq": "datetime64[ns]",
    "fyearq": "int16",
    "fqtr": "int16",
    "atq": "float64",
    "ltq": "float64",
    "seqq": "float64",
    "ceqq": "float64",
    "revtq": "float64",
    "niq": "float64",
    "ibq": "float64",
    "oiadpq": "float64",
    "cheq": "float64",
    "dlttq": "float64",
    "dlcq": "float64",
}

LINKING_DTYPES: dict[str, str] = {
    "gvkey": "str",
    "permno": "int64",
    "linkdt": "datetime64[ns]",
    "linkenddt": "datetime64[ns]",
    "linktype": "str",
    "linkprim": "str",
}
