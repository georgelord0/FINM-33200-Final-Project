"""Feature builders for the src pipeline.

Public functions are loaded lazily to keep module execution with `python -m`
predictable.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_excess_return_target": ("src.features.build_targets", "build_excess_return_target"),
    "build_direction_label": ("src.features.build_targets", "build_direction_label"),
    "build_lagged_returns": ("src.features.build_returns", "build_lagged_returns"),
    "build_cumulative_returns": ("src.features.build_returns", "build_cumulative_returns"),
    "build_log_returns": ("src.features.build_returns", "build_log_returns"),
    "build_momentum": ("src.features.build_momentum", "build_momentum"),
    "build_realized_vol": ("src.features.build_volatility", "build_realized_vol"),
    "build_downside_vol": ("src.features.build_volatility", "build_downside_vol"),
    "build_beta": ("src.features.build_volatility", "build_beta"),
    "build_turnover": ("src.features.build_liquidity", "build_turnover"),
    "build_amihud": ("src.features.build_liquidity", "build_amihud"),
    "build_dollar_volume": ("src.features.build_liquidity", "build_dollar_volume"),
    "build_volume_zscore": ("src.features.build_liquidity", "build_volume_zscore"),
    "build_book_to_market": ("src.features.build_fundamentals", "build_book_to_market"),
    "build_profitability": ("src.features.build_fundamentals", "build_profitability"),
    "build_leverage": ("src.features.build_fundamentals", "build_leverage"),
    "build_investment": ("src.features.build_fundamentals", "build_investment"),
    "build_yield_spread": ("src.features.build_macro_features", "build_yield_spread"),
    "build_inflation_change": ("src.features.build_macro_features", "build_inflation_change"),
    "build_unemployment_change": ("src.features.build_macro_features", "build_unemployment_change"),
    "build_macro_regime": ("src.features.build_macro_features", "build_macro_regime"),
    "build_market_relative_return": ("src.features.build_cross_sectional", "build_market_relative_return"),
    "build_sector_relative_return": ("src.features.build_cross_sectional", "build_sector_relative_return"),
    "build_rolling_rank": ("src.features.build_cross_sectional", "build_rolling_rank"),
    "build_cross_sectional_zscore": ("src.features.build_cross_sectional", "build_cross_sectional_zscore"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr = _EXPORTS[name]
    return getattr(import_module(module_name), attr)
