from __future__ import annotations

import argparse

import torch

from implementations.eigen_rcr_dfl.src.model import (
    EigenRCRDFL,
)
from implementations.eigen_rcr_dfl.src.oracle_cache import (
    load_or_build_oracle_cache,
)
from implementations.rank_tail_rcr_dfl.src.cache_config import (
    build_oracle_cache_config,
)
from implementations.rank_tail_rcr_dfl.src.dataset import (
    RankTailRCRDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prebuild Rank-Tail risk "
            "and oracle caches"
        )
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )

    parser.add_argument(
        "--date-column",
        default="Date",
    )

    parser.add_argument(
        "--return-type",
        choices=["simple", "log"],
        default="simple",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--market-mode",
        choices=["equal_weight", "external"],
        default="equal_weight",
    )

    parser.add_argument(
        "--market-price-csv",
        default=None,
    )

    parser.add_argument(
        "--market-column",
        default=None,
    )

    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--no-capm-intercept",
        action="store_true",
    )

    parser.add_argument(
        "--tail-q",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--tail-side",
        choices=["down", "up"],
        default="down",
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[
            25.0,
            50.0,
            100.0,
            200.0,
        ],
    )

    parser.add_argument(
        "--covariance-jitter",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--effective-jitter",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
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
            "rank_tail_rcr_dfl/"
            "cache/risk_matrix"
        ),
    )

    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
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

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )

    return parser.parse_args()


def resolve_device(name: str):
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    print("=" * 72)
    print("BUILD RANK-TAIL RISK CACHE")
    print("=" * 72)

    dataset = RankTailRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type=args.return_type,
        covariance_jitter=(
            args.covariance_jitter
        ),
        market_mode=args.market_mode,
        market_price_csv=(
            args.market_price_csv
        ),
        market_column=args.market_column,
        risk_free_rate=args.risk_free_rate,
        fit_intercept=(
            not args.no_capm_intercept
        ),
        tail_q=args.tail_q,
        tail_side=args.tail_side,
        risk_cache_dir=args.risk_cache_dir,
        force_rebuild_risk_cache=(
            args.force_rebuild_risk_cache
        ),
        dtype=torch.float64,
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

    print("\n" + "=" * 72)
    print("BUILD LAMBDA-WISE ORACLE CACHES")
    print("=" * 72)

    for risk_aversion in args.lambdas:

        model = EigenRCRDFL(
            n_assets=dataset.n_assets,
            lookback=args.lookback,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            risk_aversion=risk_aversion,
            max_weight=args.max_weight,
            eta=0.0,
            effective_jitter=(
                args.effective_jitter
            ),
        ).to(device)

        config = build_oracle_cache_config(
            dataset=dataset,
            price_csv=args.price_csv,
            date_column=args.date_column,
            lookback=args.lookback,
            return_type=args.return_type,
            covariance_jitter=(
                args.covariance_jitter
            ),
            effective_jitter=(
                args.effective_jitter
            ),
            market_mode=args.market_mode,
            market_price_csv=(
                args.market_price_csv
            ),
            market_column=(
                args.market_column
            ),
            risk_free_rate=(
                args.risk_free_rate
            ),
            fit_intercept=(
                not args.no_capm_intercept
            ),
            tail_q=args.tail_q,
            tail_side=args.tail_side,
            risk_aversion=risk_aversion,
            max_weight=args.max_weight,
        )

        (
            _,
            cache_path,
            cache_hit,
        ) = load_or_build_oracle_cache(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            cache_dir=(
                args.oracle_cache_dir
            ),
            cache_config=config,
            force_rebuild=(
                args.force_rebuild_oracle_cache
            ),
        )

        print(
            f"lambda={risk_aversion:g} | "
            f"{'HIT' if cache_hit else 'BUILD'} "
            f"| {cache_path}"
        )

    print(
        "\nALL CACHES READY"
    )


if __name__ == "__main__":
    main()