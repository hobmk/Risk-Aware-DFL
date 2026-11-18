
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser(description="RCR-DFL Covariance Condition Number Analysis")

    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--out-dir", default="implementations/final_benchmark/outputs/diagnostics/covariance_condition_numbers")
    p.add_argument("--lookback", type=int, default=60)

    p.add_argument("--start-date", default="2010-04-01")
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--end-date", default="2024-12-31")

    p.add_argument("--covariance-jitter", type=float, default=1e-6)
    p.add_argument("--ridge-shrinkage", type=float, default=0.1)
    p.add_argument("--tail-q", type=float, default=0.35)
    p.add_argument("--tail-side", choices=["down", "up"], default="down")
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--eta", type=float, default=0.10)

    p.add_argument("--market-mode", choices=["equal_weight", "external"], default="equal_weight")
    p.add_argument("--market-price-csv", default=None)
    p.add_argument("--market-column", default=None)
    p.add_argument("--risk-free-rate", type=float, default=0.0)
    p.add_argument("--no-capm-intercept", action="store_true")
    p.add_argument("--no-plot", action="store_true")

    return p.parse_args()


def eigen_diagnostics(matrices: np.ndarray):
    matrices = np.asarray(matrices, dtype=np.float64)
    matrices = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))

    eigenvalues = np.linalg.eigvalsh(matrices)

    min_eig = eigenvalues[:, 0]
    max_eig = eigenvalues[:, -1]

    condition = np.full(len(min_eig), np.inf, dtype=np.float64)

    valid = (
        (min_eig > 0)
        & np.isfinite(min_eig)
        & np.isfinite(max_eig)
    )

    condition[valid] = max_eig[valid] / min_eig[valid]

    return min_eig, max_eig, condition


def finite_quantile(series, q):
    values = series.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(np.quantile(values, q))


def summarize_conditions(frame: pd.DataFrame):
    rows = []

    for (split, model), group in frame.groupby(
        ["split", "model"], sort=False
    ):
        rows.append({
            "split": split,
            "model": model,
            "n_windows": len(group),
            "invalid_condition_count": int(
                (~np.isfinite(group["condition_number"])).sum()
            ),
            "non_positive_min_eigenvalues": int(
                (group["min_eigenvalue"] <= 0).sum()
            ),
            "min_eig_median": finite_quantile(
                group["min_eigenvalue"], 0.5
            ),
            "max_eig_median": finite_quantile(
                group["max_eigenvalue"], 0.5
            ),
            "condition_median": finite_quantile(
                group["condition_number"], 0.5
            ),
            "condition_p90": finite_quantile(
                group["condition_number"], 0.90
            ),
            "condition_p99": finite_quantile(
                group["condition_number"], 0.99
            ),
            "condition_max": finite_quantile(
                group["condition_number"], 1.0
            ),
            "worst_condition_date": group.loc[
                group["condition_number"].idxmax(), "date"
            ],
        })

    return pd.DataFrame(rows)


def check_dataset_alignment(reference, datasets):
    reference_dates = pd.DatetimeIndex(reference.target_dates)

    for name, dataset in datasets.items():
        if dataset.n_assets != reference.n_assets:
            raise RuntimeError(f"{name}: 자산 수가 다릅니다.")

        if dataset.tickers != reference.tickers:
            raise RuntimeError(f"{name}: 종목 순서가 다릅니다.")

        if not reference_dates.equals(
            pd.DatetimeIndex(dataset.target_dates)
        ):
            raise RuntimeError(f"{name}: Target 날짜가 일치하지 않습니다.")


def create_plot(per_date, models, train_end, validation_end, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))

    for model in models:
        subset = per_date.loc[
            per_date["model"] == model,
            ["date", "log10_condition"],
        ].copy()

        subset["date"] = pd.to_datetime(subset["date"])
        subset = subset.sort_values("date")

        smoothed = subset["log10_condition"].rolling(
            window=20, min_periods=5
        ).median()

        ax.plot(
            subset["date"],
            smoothed,
            label=model,
            linewidth=1.3,
        )

    ax.axvline(
        pd.Timestamp(train_end),
        linestyle="--",
        linewidth=0.8,
    )

    ax.axvline(
        pd.Timestamp(validation_end),
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_title("Rolling Covariance Condition Numbers")
    ax.set_xlabel("Date")
    ax.set_ylabel("log10(Condition Number)")

    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    price_path = Path(args.price_csv)

    if not price_path.exists():
        raise FileNotFoundError(
            f"가격 데이터가 존재하지 않습니다: {price_path}"
        )

    if not (
        args.start_date <= args.train_end
        < args.validation_end <= args.end_date
    ):
        raise ValueError("데이터 분석 날짜 범위를 확인하세요.")

    if args.eta < 0:
        raise ValueError("eta는 0 이상이어야 합니다.")

    from implementations.dfl_mvo_lee2025.src.dataset import (
        RollingMVODataset,
    )
    from implementations.covariance_baselines.src.dataset import (
        CovarianceBaselineDataset,
    )
    from implementations.rank_tail_rcr_dfl.src.dataset import (
        RankTailRCRDataset,
    )
    from implementations.eigen_rcr_dfl.src.dataset import (
        EigenRCRDataset,
    )

    common = {
        "price_csv": args.price_csv,
        "lookback": args.lookback,
        "date_column": "Date",
        "return_type": "simple",
        "covariance_jitter": args.covariance_jitter,
        "dtype": torch.float64,
    }

    market = {
        "market_mode": args.market_mode,
        "market_price_csv": args.market_price_csv,
        "market_column": args.market_column,
        "risk_free_rate": args.risk_free_rate,
        "fit_intercept": not args.no_capm_intercept,
    }

    print("\n[1/5] Loading Sample Covariance")

    sample = RollingMVODataset(
        **common,
        risk_cache_dir=(
            "implementations/dfl_mvo_lee2025/cache/risk_matrix"
        ),
    )

    print("\n[2/5] Loading Ridge Covariance")

    ridge = CovarianceBaselineDataset(
        **common,
        method="ridge",
        ridge_shrinkage=args.ridge_shrinkage,
    )

    print("\n[3/5] Loading Ledoit-Wolf Covariance")

    ledoit_wolf = CovarianceBaselineDataset(
        **common,
        method="ledoit_wolf",
    )

    print("\n[4/5] Loading Rank-Tail Covariance")

    rank_tail = RankTailRCRDataset(
        **common,
        **market,
        tail_q=args.tail_q,
        tail_side=args.tail_side,
    )

    print("\n[5/5] Loading Top-K Covariance")

    top_k = EigenRCRDataset(
        **common,
        **market,
        eigen_k=args.top_k,
        scale_mode="none",
    )

    datasets = {
        "Sample": sample,
        "Ridge": ridge,
        "Ledoit-Wolf": ledoit_wolf,
        "Rank-Tail": rank_tail,
        "Top-K": top_k,
    }

    check_dataset_alignment(sample, datasets)

    models = list(datasets.keys())
    dates = pd.DatetimeIndex(sample.target_dates)

    arrays = {
        "Sample": sample.covariances.detach().cpu().numpy(),
        "Ridge": ridge.covariances.detach().cpu().numpy(),
        "Ledoit-Wolf": ledoit_wolf.covariances.detach().cpu().numpy(),
        "Rank-Tail": rank_tail.covariances.detach().cpu().numpy(),
        "Top-K": (
            top_k.covariances + args.eta * top_k.eigen_risk
        ).detach().cpu().numpy(),
    }

    # Top-K의 기본 Sample Covariance와 DFL-MVO 공분산 동등성 확인
    topk_base = top_k.covariances.detach().cpu().numpy()

    baseline_diff = float(
        np.max(np.abs(arrays["Sample"] - topk_base))
    )

    baseline_equal = np.allclose(
        arrays["Sample"],
        topk_base,
        rtol=1e-8,
        atol=1e-12,
    )

    if not baseline_equal:
        raise RuntimeError(
            "Sample과 Top-K의 기본 공분산이 다릅니다. "
            f"Max Abs Difference={baseline_diff:.6e}"
        )

    print(
        f"\n[CHECK] Sample vs Top-K Base Covariance: "
        f"{baseline_diff:.6e}"
    )

    # 분석 기간 및 Split 설정
    selected = (
        (dates >= pd.Timestamp(args.start_date))
        & (dates <= pd.Timestamp(args.end_date))
    )

    indices = np.flatnonzero(selected)

    if len(indices) == 0:
        raise RuntimeError("선택한 기간에 분석할 데이터가 없습니다.")

    splits = np.full(len(dates), "outside", dtype=object)

    splits[
        selected & (dates <= pd.Timestamp(args.train_end))
    ] = "train"

    splits[
        selected
        & (dates > pd.Timestamp(args.train_end))
        & (dates <= pd.Timestamp(args.validation_end))
    ] = "validation"

    splits[
        selected & (dates > pd.Timestamp(args.validation_end))
    ] = "test"

    # 모델별 고유값 및 조건수 계산
    frames = []

    for name in models:
        print(f"[ANALYZE] {name}")

        min_eig, max_eig, condition = eigen_diagnostics(
            arrays[name][indices]
        )

        log_condition = np.full(len(condition), np.nan)

        valid = (
            np.isfinite(condition)
            & (condition > 0)
        )

        log_condition[valid] = np.log10(condition[valid])

        frame = pd.DataFrame({
            "date": dates[indices].strftime("%Y-%m-%d"),
            "split": splits[indices],
            "model": name,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": condition,
            "log10_condition": log_condition,
        })

        frames.append(frame)

    per_date = pd.concat(frames, ignore_index=True)
    summary = summarize_conditions(per_date)

    summary["split"] = pd.Categorical(
        summary["split"],
        categories=["train", "validation", "test"],
        ordered=True,
    )

    summary["model"] = pd.Categorical(
        summary["model"],
        categories=models,
        ordered=True,
    )

    summary = summary.sort_values(
        ["split", "model"]
    ).reset_index(drop=True)

    summary["split"] = summary["split"].astype(str)
    summary["model"] = summary["model"].astype(str)

    # 결과 저장
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_date.to_csv(
        out_dir / "condition_per_date.csv",
        index=False,
        float_format="%.12g",
    )

    summary.to_csv(
        out_dir / "condition_summary.csv",
        index=False,
        float_format="%.12g",
    )

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "n_assets": sample.n_assets,
        "tickers": sample.tickers,
        "n_windows": len(indices),
        "sample_topk_base_max_abs_diff": baseline_diff,
        "condition_definition": "lambda_max / lambda_min",
        "non_positive_min_eigenvalue": "condition=inf",
    }

    with open(
        out_dir / "analysis_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 콘솔 출력
    print("\n" + "=" * 110)
    print("COVARIANCE CONDITION NUMBER SUMMARY")
    print("=" * 110)

    with pd.option_context(
        "display.width", 180,
        "display.max_rows", 50,
        "display.float_format", lambda x: f"{x:.6g}",
    ):
        print(summary.to_string(index=False))

    # 조건수 시계열 그래프
    if not args.no_plot:
        try:
            create_plot(
                per_date=per_date,
                models=models,
                train_end=args.train_end,
                validation_end=args.validation_end,
                path=out_dir / "condition_timeseries.png",
            )

        except ImportError:
            print("[INFO] matplotlib 미설치: 그래프 생성 생략")

    print("\n" + "=" * 110)
    print("ANALYSIS COMPLETED")
    print("=" * 110)

    print(f"Assets: {sample.n_assets}")
    print(f"Windows: {len(indices)}")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
