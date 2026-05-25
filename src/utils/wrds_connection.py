"""Singleton WRDS connection manager with retry logic.

Provides a thread-safe, auto-reconnecting connection to the WRDS PostgreSQL
database.  Credentials are read from a ``.env`` file via *python-dotenv*.

Typical usage
-------------
::

    from src.utils.wrds_connection import get_connection

    with get_connection() as wrds:
        df = wrds.query("SELECT * FROM crsp.msf LIMIT 10")
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Load .env at import time so credentials are available early.
load_dotenv()


# ---------------------------------------------------------------------------
# Singleton connection
# ---------------------------------------------------------------------------

_instance: WRDSConnection | None = None


class WRDSConnection:
    """Manages a single WRDS database connection with auto-reconnect.

    Do **not** instantiate directly -- use :func:`get_connection` to obtain
    the module-level singleton.

    Parameters
    ----------
    username:
        WRDS username.  Defaults to the ``WRDS_USERNAME`` environment variable.
    """

    def __init__(self, username: str | None = None, password: str | None = None) -> None:
        self._username: str = username or os.environ.get("WRDS_USERNAME", "")
        self._password: str = password or os.environ.get("WRDS_PASSWORD", "")
        if not self._username:
            raise EnvironmentError(
                "WRDS_USERNAME not set.  Add it to your .env file or pass it explicitly."
            )
        if not self._password:
            raise EnvironmentError(
                "WRDS_PASSWORD not set. Add it to your .env file or pass it explicitly. "
                "If you prefer pgpass/keyring, use the wrds package setup outside "
                "scripts/reproduce.py."
            )
        self._conn: Any | None = None
        log.info("WRDSConnection created for user '{}'", self._username)

    # -- connection lifecycle ------------------------------------------------

    def _connect(self) -> Any:
        """Establish (or re-establish) the underlying WRDS connection."""
        log.info("Connecting to WRDS as '{}' ...", self._username)
        try:
            import wrds
        except ImportError as e:
            raise ImportError(
                "wrds package not installed. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from e
        try:
            conn = wrds.Connection(
                wrds_username=self._username,
                wrds_password=self._password,
            )
        except Exception as exc:
            raise RuntimeError(_wrds_auth_message(exc)) from exc
        log.info("WRDS connection established.")
        return conn

    def _ensure_alive(self) -> Any:
        """Return a live connection, reconnecting if the existing one is stale."""
        if self._conn is None:
            self._conn = self._connect()
            return self._conn

        # Ping the connection to check liveness.
        try:
            self._conn.raw_sql("SELECT 1")
        except Exception:
            log.warning("Stale WRDS connection detected -- reconnecting.")
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = self._connect()

        return self._conn

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> WRDSConnection:
        self._ensure_alive()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
                log.info("WRDS connection closed.")
            except Exception as exc:
                log.warning("Error closing WRDS connection: {}", exc)
            finally:
                self._conn = None

    # -- query methods -------------------------------------------------------

    def query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        date_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return the results as a DataFrame.

        Parameters
        ----------
        sql:
            SQL query string.  Use ``%(name)s`` style placeholders for
            parameters.
        params:
            Optional mapping of parameter names to values.
        date_cols:
            Column names to parse as datetimes after loading.

        Returns
        -------
        pd.DataFrame
            Query results.
        """
        try:
            conn = self._ensure_alive()
            log.debug("Executing query ({} chars): {}...", len(sql), sql[:120])
            df: pd.DataFrame = conn.raw_sql(sql, params=params, date_cols=date_cols)
        except RuntimeError:
            raise
        except Exception as exc:
            if _looks_like_auth_error(exc):
                raise RuntimeError(_wrds_auth_message(exc)) from exc
            raise

        log.info("Query returned {} rows, {} columns.", len(df), len(df.columns))
        return df

    def chunked_query(
        self,
        sql_template: str,
        start_year: int,
        end_year: int,
        params: dict[str, Any] | None = None,
        date_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Execute a query template once per year and concatenate results.

        The *sql_template* must contain a ``{year}`` placeholder that will
        be formatted with each year in ``[start_year, end_year]``.

        Parameters
        ----------
        sql_template:
            SQL string with a ``{year}`` placeholder.
        start_year:
            First year (inclusive).
        end_year:
            Last year (inclusive).
        params:
            Additional query parameters forwarded to :meth:`query`.
        date_cols:
            Columns to parse as datetimes.

        Returns
        -------
        pd.DataFrame
            Concatenation of all per-year results.
        """
        frames: list[pd.DataFrame] = []
        years = range(start_year, end_year + 1)
        log.info(
            "Running chunked query for years {}-{} ({} chunks)",
            start_year,
            end_year,
            len(years),
        )

        for year in years:
            sql = sql_template.format(year=year)
            log.debug("Chunked query: year={}", year)
            df = self.query(sql, params=params, date_cols=date_cols)
            if not df.empty:
                frames.append(df)
            log.debug("Year {} returned {} rows.", year, len(df))

        if not frames:
            log.warning("Chunked query returned no data for years {}-{}", start_year, end_year)
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        log.info(
            "Chunked query complete: {} total rows from {} years.",
            len(result),
            len(frames),
        )
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_connection(username: str | None = None, password: str | None = None) -> WRDSConnection:
    """Return the module-level singleton :class:`WRDSConnection`.

    On the first call the connection object is created (but the actual TCP
    connection is deferred until :meth:`WRDSConnection.query` or entering the
    context manager).

    Parameters
    ----------
    username:
        Override WRDS username.  Only used on the first call.
    password:
        Override WRDS password.  Only used on the first call.

    Returns
    -------
    WRDSConnection
        The singleton instance.
    """
    global _instance  # noqa: PLW0603

    if _instance is None:
        _instance = WRDSConnection(username=username, password=password)
        log.info("Singleton WRDSConnection initialised.")

    return _instance


def _looks_like_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = (
        "pam authentication failed",
        "ldap authentication failed",
        "password authentication failed",
        "no password supplied",
    )
    return any(n in text for n in needles)


def _wrds_auth_message(exc: Exception) -> str:
    detail = str(exc).strip()
    return (
        "WRDS authentication failed. scripts/reproduce.py uses WRDS_USERNAME "
        "and WRDS_PASSWORD from .env. If a Duo push appears, approve it while "
        "scripts/reproduce.py keeps polling. Also verify the password in .env "
        "is your normal WRDS password.\n\n"
        f"Original WRDS error: {detail}"
    )
