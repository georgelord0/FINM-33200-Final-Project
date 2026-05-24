"""Feature builders for the src pipeline.

Each sub-module exposes a ``build()`` orchestrator that loads raw data,
constructs features, and persists the results to ``data/features/``.
"""

from src.features.build_cross_sectional import (
    build_cross_sectional_zscore,
    build_market_relative_return,
    build_rolling_rank,
    build_sector_relative_return,
)
from src.features.build_fundamentals import (
    build_book_to_market,
    build_investment,
    build_leverage,
    build_profitability,
)
from src.features.build_liquidity import (
    build_amihud,
    build_dollar_volume,
    build_turnover,
    build_volume_zscore,
)
from src.features.build_macro_features import (
    build_inflation_change,
    build_macro_regime,
    build_unemployment_change,
    build_yield_spread,
)
from src.features.build_momentum import build_momentum
from src.features.build_returns import (
    build_cumulative_returns,
    build_lagged_returns,
    build_log_returns,
)
from src.features.build_targets import (
    build_direction_label,
    build_excess_return_target,
)
from src.features.build_volatility import (
    build_beta,
    build_downside_vol,
    build_realized_vol,
)

__all__ = [
    # targets
    "build_excess_return_target",
    "build_direction_label",
    # returns
    "build_lagged_returns",
    "build_cumulative_returns",
    "build_log_returns",
    # momentum
    "build_momentum",
    # volatility
    "build_realized_vol",
    "build_downside_vol",
    "build_beta",
    # liquidity
    "build_turnover",
    "build_amihud",
    "build_dollar_volume",
    "build_volume_zscore",
    # fundamentals
    "build_book_to_market",
    "build_profitability",
    "build_leverage",
    "build_investment",
    # macro
    "build_yield_spread",
    "build_inflation_change",
    "build_unemployment_change",
    "build_macro_regime",
    # cross-sectional
    "build_market_relative_return",
    "build_sector_relative_return",
    "build_rolling_rank",
    "build_cross_sectional_zscore",
]
