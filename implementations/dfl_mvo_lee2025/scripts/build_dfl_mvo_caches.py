from __future__ import annotations

import argparse
from pathlib import Path

import torch

from implementations.dfl_mvo_lee2025.src.cache import (
    load_or_build_oracle_cache,
)

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--price-csv",
        default=(
            "data/raw/"
            "dow30_adjusted_close.csv"
        ),
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[
            25,
            50,
            100,
            200,
        ],
    )

    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--risk-cache-dir",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "cache/risk_matrix"
        ),
    )

    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "cache/oracle"
        ),
    )

    parser.add_argument(
        "--force-rebuild-risk-cache",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-oracle-cache",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print(
        "BUILD DFL-MVO "
        "COVARIANCE CACHE"
    )
    print("=" * 72)

    dataset = RollingMVODataset(
        price_csv=Path(
            args.price_csv
        ),
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float32,
        risk_cache_dir=(
            args.risk_cache_dir
        ),
        force_rebuild_risk_cache=(
            args.force_rebuild_risk_cache
        ),
    )

    print(
        "Risk cache : "
        f"{'HIT' if dataset.risk_cache_hit else 'BUILD'}"
    )

    print(
        f"Risk key   : "
        f"{dataset.risk_cache_key}"
    )

    print(
        f"Risk path  : "
        f"{dataset.risk_cache_path}"
    )

    print(
        f"N samples  : "
        f"{len(dataset)}"
    )

    print(
        f"N assets   : "
        f"{dataset.n_assets}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "BUILD LAMBDA-WISE "
        "ORACLE CACHES"
    )

    print("=" * 72)

    for risk_aversion in args.lambdas:

        _, path, hit = (
            load_or_build_oracle_cache(
                dataset=dataset,
                risk_aversion=(
                    risk_aversion
                ),
                max_weight=(
                    args.max_weight
                ),
                batch_size=(
                    args.batch_size
                ),
                cache_dir=(
                    args.oracle_cache_dir
                ),
                force_rebuild=(
                    args.force_rebuild_oracle_cache
                ),
            )
        )

        print(
            f"lambda="
            f"{risk_aversion:g} "
            f"| "
            f"{'HIT' if hit else 'BUILD'} "
            f"| {path}"
        )

    print(
        "\nALL DFL-MVO "
        "CACHES READY"
    )


if __name__ == "__main__":
    main()