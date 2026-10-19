from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [42, 43, 44, 45, 46]
ROOT = Path("implementations/rank_tail_rcr_dfl/outputs")

BASE_DOWN = ROOT / "q_grid/q_0.35"
BASE_UP = ROOT / "upside_placebo/q_0.35"

EXPERIMENTS = [
    ("fixed_msecoef15", 0.10, 16.6666666667,
     ROOT / "loss_balance_sensitivity/fixed_msecoef15/alpha_0.10"),
    ("fixed_msecoef15", 0.25, 20.0,
     ROOT / "loss_balance_sensitivity/fixed_msecoef15/alpha_0.25"),
    ("fixed_msecoef15", 0.50, 30.0, None),
    ("fixed_msecoef15", 0.75, 60.0,
     ROOT / "loss_balance_sensitivity/fixed_msecoef15/alpha_0.75"),

    ("mse_scale", 0.50, 5.0,
     ROOT / "loss_balance_sensitivity/mse_scale/mse_5"),
    ("mse_scale", 0.50, 10.0,
     ROOT / "loss_balance_sensitivity/mse_scale/mse_10"),
    ("mse_scale", 0.50, 30.0, None),
    ("mse_scale", 0.50, 60.0,
     ROOT / "loss_balance_sensitivity/mse_scale/mse_60"),
    ("mse_scale", 0.50, 100.0,
     ROOT / "loss_balance_sensitivity/mse_scale/mse_100"),
]


def load_matrix(path: Path, value_candidates: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)

    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        raise KeyError(f"date column 없음: {path}")

    df[date_col] = pd.to_datetime(df[date_col])

    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)

    if ticker_col is not None:
        value_col = next((c for c in value_candidates if c in df.columns), None)
        if value_col is None:
            raise KeyError(f"value column 없음: {path}")

        return (
            df.pivot(index=date_col, columns=ticker_col, values=value_col)
            .sort_index()
            .astype(float)
        )

    return (
        df.set_index(date_col)
        .sort_index()
        .astype(float)
    )


def run_dir(group: str, root: Path | None, side: str, seed: int) -> Path:
    if root is None:
        base = BASE_DOWN if side == "down" else BASE_UP
        return base / f"seed_{seed}"

    return root / side / "q_0.35" / f"seed_{seed}"


def predictor_metrics(down_dir: Path, up_dir: Path) -> dict:
    down = load_matrix(
        down_dir / "predicted_returns.csv",
        ["predicted_return"],
    )
    up = load_matrix(
        up_dir / "predicted_returns.csv",
        ["predicted_return"],
    )
    true = load_matrix(
        down_dir / "true_returns.csv",
        ["true_return"],
    )

    common_dates = down.index.intersection(up.index).intersection(true.index)
    common_cols = down.columns.intersection(up.columns).intersection(true.columns)

    d = down.loc[common_dates, common_cols].to_numpy()
    u = up.loc[common_dates, common_cols].to_numpy()
    y = true.loc[common_dates, common_cols].to_numpy()

    flat_d = d.ravel()
    flat_u = u.ravel()

    flat_corr = np.corrcoef(flat_d, flat_u)[0, 1]
    cosine = np.dot(flat_d, flat_u) / (
        np.linalg.norm(flat_d) * np.linalg.norm(flat_u)
    )

    daily_corr = []
    top5 = []

    for a, b in zip(d, u):
        daily_corr.append(np.corrcoef(a, b)[0, 1])

        ka = set(np.argsort(a)[-5:])
        kb = set(np.argsort(b)[-5:])
        top5.append(len(ka & kb) / 5.0)

    return {
        "flat_corr": flat_corr,
        "cosine": cosine,
        "daily_cs_corr": np.nanmean(daily_corr),
        "rmse_diff": np.sqrt(np.mean((d - u) ** 2)),
        "top5_overlap": np.mean(top5),
        "down_mse": np.mean((d - y) ** 2),
        "up_mse": np.mean((u - y) ** 2),
    }


def portfolio_metrics(path: Path) -> dict:
    df = pd.read_csv(path)
    row = df.iloc[0]

    candidates = {
        "annualized_return": ["annualized_return", "annualized_return_cagr", "cagr"],
        "volatility": ["volatility", "annualized_volatility"],
        "sharpe": ["sharpe", "sharpe_ratio", "sharpe_ratio_rf0"],
        "max_drawdown": ["max_drawdown", "maximum_drawdown", "mdd"],
        "annualized_turnover": ["annualized_turnover", "turnover"],
        "active_assets": ["active_assets"],
    }

    result = {}

    for name, cols in candidates.items():
        col = next((c for c in cols if c in df.columns), None)
        result[name] = float(row[col]) if col else np.nan

    return result


def main():
    rows = []

    for group, alpha, mse_scale, root in EXPERIMENTS:
        for seed in SEEDS:
            down_dir = run_dir(group, root, "down", seed)
            up_dir = run_dir(group, root, "up", seed)

            similarity = predictor_metrics(down_dir, up_dir)
            down_perf = portfolio_metrics(down_dir / "portfolio_summary.csv")
            up_perf = portfolio_metrics(up_dir / "portfolio_summary.csv")

            rows.append({
                "group": group,
                "alpha": alpha,
                "mse_scale": mse_scale,
                "mse_coefficient": (1.0 - alpha) * mse_scale,
                "seed": seed,
                **similarity,
                **{f"down_{k}": v for k, v in down_perf.items()},
                **{f"up_{k}": v for k, v in up_perf.items()},
            })

    raw = pd.DataFrame(rows)

    output_dir = ROOT / "loss_balance_sensitivity/summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(output_dir / "all_runs.csv", index=False)

    metrics = [
        "flat_corr",
        "daily_cs_corr",
        "rmse_diff",
        "top5_overlap",
        "down_mse",
        "up_mse",
        "down_sharpe",
        "up_sharpe",
        "down_volatility",
        "up_volatility",
        "down_max_drawdown",
        "up_max_drawdown",
    ]

    for group in ["fixed_msecoef15", "mse_scale"]:
        subset = raw[raw["group"] == group]

        summary = (
            subset.groupby(["alpha", "mse_scale", "mse_coefficient"])[metrics]
            .agg(["mean", "std"])
        )

        summary.to_csv(output_dir / f"{group}_summary.csv")

        print()
        print("=" * 120)
        print(group.upper())
        print("=" * 120)

        compact = (
            subset.groupby(["alpha", "mse_scale", "mse_coefficient"])
            .agg(
                flat_corr=("flat_corr", "mean"),
                daily_cs_corr=("daily_cs_corr", "mean"),
                rmse_diff=("rmse_diff", "mean"),
                top5_overlap=("top5_overlap", "mean"),
                down_mse=("down_mse", "mean"),
                up_mse=("up_mse", "mean"),
                down_sharpe=("down_sharpe", "mean"),
                up_sharpe=("up_sharpe", "mean"),
            )
            .reset_index()
        )

        print(compact.to_string(index=False))

    print()
    print("Saved:")
    print(output_dir)


if __name__ == "__main__":
    main()