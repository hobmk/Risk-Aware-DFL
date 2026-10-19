from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--validation-end", default="2022-12-31")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])

    parser.add_argument(
        "--down-root",
        default="implementations/rank_tail_rcr_dfl/outputs/q_grid/q_0.35",
    )
    parser.add_argument(
        "--up-root",
        default="implementations/rank_tail_rcr_dfl/outputs/upside_placebo/q_0.35",
    )
    parser.add_argument(
        "--baseline-root",
        default=(
            "implementations/dfl_mvo_lee2025/outputs/combined/dow30/"
            "final_full_grid_h64_d0_s30_std/alpha_0.50/lambda_1.00/maxw_0.20"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/rank_tail_rcr_dfl/outputs/mechanism/"
            "q_0.35/down_up_similarity"
        ),
    )
    return parser.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def offdiag(matrix: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    idx = np.arange(matrix.shape[-1])
    result[..., idx, idx] = 0.0
    return result


def positive_offdiag(matrix: np.ndarray) -> np.ndarray:
    return np.maximum(offdiag(matrix), 0.0)


def quadratic_exposure(
    weights: np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def load_wide_weights(
    path: Path,
    tickers: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    return (
        df.set_index("date")
        .sort_index()
        .reindex(columns=tickers)
        .astype(np.float64)
    )


def load_baseline_weights(
    path: Path,
    tickers: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    return (
        df.pivot(
            index="date",
            columns="ticker",
            values="predicted_weight",
        )
        .sort_index()
        .reindex(columns=tickers)
        .astype(np.float64)
    )


def build_dataset(
    tail_q: float,
    tail_side: str,
) -> RankTailRCRDataset:
    return RankTailRCRDataset(
        price_csv="data/raw/dow30_adjusted_close.csv",
        lookback=60,
        date_column="Date",
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        tail_q=tail_q,
        tail_side=tail_side,
        dtype=torch.float64,
    )


def matrix_diagnostics(
    dates: pd.DatetimeIndex,
    down_rt: np.ndarray,
    up_rt: np.ndarray,
    down_eff: np.ndarray,
    up_eff: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for i, date in enumerate(dates):
        dr = down_rt[i]
        ur = up_rt[i]
        de = down_eff[i]
        ue = up_eff[i]

        dr_off = offdiag(dr)
        ur_off = offdiag(ur)
        de_off = offdiag(de)
        ue_off = offdiag(ue)

        rows.append({
            "date": date,

            "rt_full_cosine":
                cosine(dr.ravel(), ur.ravel()),

            "rt_offdiag_cosine":
                cosine(dr_off.ravel(), ur_off.ravel()),

            "rt_offdiag_corr":
                correlation(dr_off.ravel(), ur_off.ravel()),

            "rt_relative_fro_difference":
                np.linalg.norm(dr - ur, ord="fro")
                / np.linalg.norm(dr, ord="fro"),

            "rt_offdiag_relative_fro_difference":
                np.linalg.norm(dr_off - ur_off, ord="fro")
                / np.linalg.norm(dr_off, ord="fro"),

            "effective_full_cosine":
                cosine(de.ravel(), ue.ravel()),

            "effective_offdiag_cosine":
                cosine(de_off.ravel(), ue_off.ravel()),

            "effective_offdiag_corr":
                correlation(de_off.ravel(), ue_off.ravel()),

            "effective_relative_fro_difference":
                np.linalg.norm(de - ue, ord="fro")
                / np.linalg.norm(de, ord="fro"),
        })

    return pd.DataFrame(rows)


def weight_diagnostics(
    down: pd.DataFrame,
    up: pd.DataFrame,
) -> pd.DataFrame:
    common = down.index.intersection(up.index)
    down = down.loc[common].to_numpy()
    up = up.loc[common].to_numpy()

    rows = []

    for i, date in enumerate(common):
        d = down[i]
        u = up[i]

        top5_d = set(np.argsort(d)[-5:])
        top5_u = set(np.argsort(u)[-5:])

        rows.append({
            "date": date,
            "weight_cosine": cosine(d, u),
            "weight_corr": correlation(d, u),
            "weight_l1": np.abs(d - u).sum(),
            "weight_max_abs_diff": np.abs(d - u).max(),
            "top5_overlap": len(top5_d & top5_u) / 5.0,
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("DOWNSIDE vs UPSIDE RANK-TAIL SIMILARITY DIAGNOSTIC")
    print("=" * 100)

    down_dataset = build_dataset(
        args.tail_q,
        "down",
    )
    up_dataset = build_dataset(
        args.tail_q,
        "up",
    )

    dates = pd.DatetimeIndex(
        down_dataset.target_dates
    ).normalize()

    mask = dates > pd.Timestamp(
        args.validation_end
    )

    test_dates = dates[mask]

    down_rt = (
        down_dataset.rank_tail_risk[mask]
        .cpu()
        .numpy()
    )
    up_rt = (
        up_dataset.rank_tail_risk[mask]
        .cpu()
        .numpy()
    )

    down_eff = (
        down_dataset.covariances[mask]
        .cpu()
        .numpy()
    )
    up_eff = (
        up_dataset.covariances[mask]
        .cpu()
        .numpy()
    )

    tickers = list(
        down_dataset.tickers
    )

    # ---------------------------------------------------------
    # 1. Matrix similarity
    # ---------------------------------------------------------

    matrix_df = matrix_diagnostics(
        dates=test_dates,
        down_rt=down_rt,
        up_rt=up_rt,
        down_eff=down_eff,
        up_eff=up_eff,
    )

    matrix_df.to_csv(
        output_dir / "matrix_similarity_daily.csv",
        index=False,
    )

    matrix_summary = (
        matrix_df
        .drop(columns="date")
        .agg(["mean", "std", "min", "max"])
        .T
    )

    matrix_summary.to_csv(
        output_dir / "matrix_similarity_summary.csv"
    )

    # ---------------------------------------------------------
    # 2. Exposure similarity using SAME baseline DFL weights
    # ---------------------------------------------------------

    down_positive = positive_offdiag(
        down_rt
    )
    up_positive = positive_offdiag(
        up_rt
    )

    exposure_rows = []

    for seed in [
        s for s in args.seeds
        if s in {42, 43, 44}
    ]:
        baseline_path = (
            Path(args.baseline_root)
            / f"seed_{seed}"
            / "asset_predictions.csv"
        )

        weights = load_baseline_weights(
            baseline_path,
            tickers,
        )

        common = test_dates.intersection(
            weights.index
        )

        positions = test_dates.get_indexer(
            common
        )

        w = weights.loc[
            common
        ].to_numpy()

        down_exp = quadratic_exposure(
            w,
            down_positive[positions],
        )

        up_exp = quadratic_exposure(
            w,
            up_positive[positions],
        )

        exposure_rows.append({
            "seed": seed,
            "n_dates": len(common),
            "down_mean_exposure": down_exp.mean(),
            "up_mean_exposure": up_exp.mean(),
            "down_up_exposure_corr": correlation(
                down_exp,
                up_exp,
            ),
        })

    exposure_df = pd.DataFrame(
        exposure_rows
    )

    exposure_df.to_csv(
        output_dir / "exposure_similarity_by_seed.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Downside vs Upside learned portfolio weights
    # ---------------------------------------------------------

    weight_rows = []
    daily_weight_frames = []

    for seed in args.seeds:
        down_path = (
            Path(args.down_root)
            / f"seed_{seed}"
            / "portfolio_weights.csv"
        )

        up_path = (
            Path(args.up_root)
            / f"seed_{seed}"
            / "portfolio_weights.csv"
        )

        down_weights = load_wide_weights(
            down_path,
            tickers,
        )

        up_weights = load_wide_weights(
            up_path,
            tickers,
        )

        daily = weight_diagnostics(
            down_weights,
            up_weights,
        )

        daily.insert(
            0,
            "seed",
            seed,
        )

        daily_weight_frames.append(
            daily
        )

        weight_rows.append({
            "seed": seed,
            "n_dates": len(daily),
            "mean_weight_cosine":
                daily["weight_cosine"].mean(),
            "mean_weight_corr":
                daily["weight_corr"].mean(),
            "mean_weight_l1":
                daily["weight_l1"].mean(),
            "mean_weight_max_abs_diff":
                daily["weight_max_abs_diff"].mean(),
            "mean_top5_overlap":
                daily["top5_overlap"].mean(),
        })

    weight_df = pd.DataFrame(
        weight_rows
    )

    weight_daily = pd.concat(
        daily_weight_frames,
        ignore_index=True,
    )

    weight_df.to_csv(
        output_dir / "weight_similarity_by_seed.csv",
        index=False,
    )

    weight_daily.to_csv(
        output_dir / "weight_similarity_daily.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("MATRIX SIMILARITY")
    print("=" * 100)

    display_matrix = [
        "rt_full_cosine",
        "rt_offdiag_cosine",
        "rt_offdiag_corr",
        "rt_relative_fro_difference",
        "rt_offdiag_relative_fro_difference",
        "effective_full_cosine",
        "effective_offdiag_cosine",
        "effective_offdiag_corr",
        "effective_relative_fro_difference",
    ]

    print(
        matrix_summary.loc[
            display_matrix,
            ["mean", "std"],
        ].to_string()
    )

    print()
    print("=" * 100)
    print("POSITIVE-OFFDIAGONAL EXPOSURE SIMILARITY")
    print("=" * 100)

    print(
        exposure_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("DOWNSIDE vs UPSIDE PORTFOLIO WEIGHT SIMILARITY")
    print("=" * 100)

    print(
        weight_df.to_string(
            index=False
        )
    )

    print()
    print("Overall weight means:")
    print(
        weight_df[
            [
                "mean_weight_cosine",
                "mean_weight_corr",
                "mean_weight_l1",
                "mean_weight_max_abs_diff",
                "mean_top5_overlap",
            ]
        ]
        .mean()
        .to_string()
    )

    print()
    print("Saved:")
    print(output_dir)


if __name__ == "__main__":
    main()