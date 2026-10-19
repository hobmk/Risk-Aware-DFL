from __future__ import annotations

from pathlib import Path
from typing import Any


def build_oracle_cache_config(
    *,
    dataset,
    price_csv: str,
    date_column: str,
    lookback: int,
    return_type: str,
    covariance_jitter: float,
    effective_jitter: float,
    market_mode: str,
    market_price_csv: str | None,
    market_column: str | None,
    risk_free_rate: float,
    fit_intercept: bool,
    tail_q: float,
    tail_side: str,
    risk_aversion: float,
    max_weight: float,
) -> dict[str, Any]:

    return {
        "cache_version": "rank_tail_oracle_v2",
        "risk_method": (
            "rank_tail_residual_dependence_v1"
        ),

        "price_csv": str(
            Path(price_csv)
        ),

        "date_column": date_column,
        "lookback": int(lookback),
        "return_type": return_type,

        "covariance_jitter": float(
            covariance_jitter
        ),

        "effective_jitter": float(
            effective_jitter
        ),

        "market_mode": market_mode,

        "market_price_csv": (
            str(Path(market_price_csv))
            if market_price_csv
            else None
        ),

        "market_column": market_column,

        "risk_free_rate": float(
            risk_free_rate
        ),

        "fit_intercept": bool(
            fit_intercept
        ),

        "tail_q": float(
            tail_q
        ),

        "tail_side": tail_side,

        "risk_aversion": float(
            risk_aversion
        ),

        "max_weight": float(
            max_weight
        ),

        "model_eta": 0.0,

        "risk_cache_key": getattr(
            dataset,
            "risk_cache_key",
            None,
        ),

        "n_assets": int(
            dataset.n_assets
        ),

        "n_samples": int(
            len(dataset)
        ),
    }