import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RETURN_COLUMNS = [
    "portfolio_return",
    "daily_return",
    "return",
    "gross_return",
    "portfolio_ret",
    "ret",
]

TURNOVER_COLUMNS = [
    "turnover",
    "portfolio_turnover",
    "daily_turnover",
]

DATE_COLUMNS = [
    "date",
    "Date",
    "datetime",
    "timestamp",
]


def find_column(df, candidates):
    columns = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    for column in df.columns:
        name = str(column).lower()
        if any(candidate.lower() in name for candidate in candidates):
            return column

    return None


def parse_config(path):
    text = str(path).replace("\\", "/")

    alpha = re.search(r"alpha_([0-9.]+)", text)
    lam = re.search(r"lambda_([0-9.]+)", text)
    seed = re.search(r"seed_(\d+)", text)

    return {
        "alpha": float(alpha.group(1)) if alpha else np.nan,
        "lambda": float(lam.group(1)) if lam else np.nan,
        "seed": int(seed.group(1)) if seed else np.nan,
    }


def max_drawdown(returns):
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return drawdown.min()


def calculate_metrics(returns):
    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {
            "cagr": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "cumulative_return": np.nan,
            "n_days": 0,
        }

    wealth = (1.0 + returns).cumprod()
    years = len(returns) / 252.0

    cumulative_return = wealth.iloc[-1] - 1.0
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0

    volatility = returns.std(ddof=1) * np.sqrt(252)

    if returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    else:
        sharpe = np.nan

    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "mdd": max_drawdown(returns),
        "cumulative_return": cumulative_return,
        "n_days": len(returns),
    }


def process_file(csv_path, input_root, output_root, costs_bps):
    df = pd.read_csv(csv_path)

    return_col = find_column(df, RETURN_COLUMNS)
    turnover_col = find_column(df, TURNOVER_COLUMNS)

    if return_col is None:
        raise ValueError(
            f"Return column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if turnover_col is None:
        raise ValueError(
            f"Turnover column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    gross_return = pd.to_numeric(
        df[return_col],
        errors="coerce"
    )

    turnover = pd.to_numeric(
        df[turnover_col],
        errors="coerce"
    ).fillna(0.0)

    valid = gross_return.notna()

    df = df.loc[valid].copy()
    gross_return = gross_return.loc[valid]
    turnover = turnover.loc[valid]

    config = parse_config(csv_path)

    # ---------------------------------------------------------
    # Transaction-cost-adjusted daily returns
    #
    # cost = turnover × transaction_cost
    # net return = gross return - cost
    # ---------------------------------------------------------
    for bps in costs_bps:
        cost_rate = bps / 10000.0
        df[f"net_return_{int(bps)}bp"] = (
            gross_return - turnover * cost_rate
        )

    # ---------------------------------------------------------
    # Save adjusted daily portfolio
    # ---------------------------------------------------------
    relative_path = csv_path.relative_to(input_root)
    output_file = output_root / relative_path

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_file, index=False)

    # ---------------------------------------------------------
    # Seed-level performance
    # ---------------------------------------------------------
    rows = []

    gross_metrics = calculate_metrics(gross_return)

    rows.append({
        **config,
        "cost_bps": 0,
        "cost_rate": 0.0,
        **gross_metrics,
        "avg_turnover": turnover.mean(),
        "total_turnover": turnover.sum(),
        "source": str(csv_path),
    })

    for bps in costs_bps:
        net_return = df[f"net_return_{int(bps)}bp"]

        metrics = calculate_metrics(net_return)

        rows.append({
            **config,
            "cost_bps": bps,
            "cost_rate": bps / 10000.0,
            **metrics,
            "avg_turnover": turnover.mean(),
            "total_turnover": turnover.sum(),
            "source": str(csv_path),
        })

    return rows


def aggregate_results(seed_results):
    metric_columns = [
        "cagr",
        "volatility",
        "sharpe",
        "mdd",
        "cumulative_return",
        "avg_turnover",
        "total_turnover",
        "n_days",
    ]

    grouped = seed_results.groupby(
        [
            "alpha",
            "lambda",
            "cost_bps",
            "cost_rate",
        ],
        dropna=False,
    )

    rows = []

    for keys, group in grouped:
        row = dict(zip(
            [
                "alpha",
                "lambda",
                "cost_bps",
                "cost_rate",
            ],
            keys,
        ))

        row["n_seeds"] = group["seed"].nunique()

        for column in metric_columns:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            )

            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Apply transaction costs to RCR-DFL portfolio results."
    )

    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing alpha/lambda/seed results.",
    )

    parser.add_argument(
        "--cost-bps",
        nargs="+",
        type=float,
        default=[5, 10],
        help="Transaction costs in basis points. Default: 5 10",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output directory.",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_dir).resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_root}"
        )

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Use --overwrite if you want to replace it."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        input_root.rglob("daily_portfolio.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No daily_portfolio.csv found under {input_root}"
        )

    print("=" * 100)
    print("TRANSACTION COST POST-PROCESSING")
    print("=" * 100)
    print(f"Input root : {input_root}")
    print(f"Output dir : {output_root}")
    print(f"Cost       : {args.cost_bps} bps")
    print(f"Files      : {len(files)}")
    print()

    all_rows = []
    failed = []

    for csv_path in files:
        try:
            rows = process_file(
                csv_path=csv_path,
                input_root=input_root,
                output_root=output_root,
                costs_bps=args.cost_bps,
            )

            all_rows.extend(rows)

        except Exception as e:
            failed.append({
                "file": str(csv_path),
                "error": str(e),
            })

    if not all_rows:
        raise RuntimeError(
            "No files were successfully processed."
        )

    # ---------------------------------------------------------
    # Seed-level result
    # ---------------------------------------------------------
    seed_results = pd.DataFrame(all_rows)

    seed_results.to_csv(
        output_root / "transaction_cost_seed_results.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Alpha × Lambda × Cost aggregated result
    # ---------------------------------------------------------
    grid_summary = aggregate_results(
        seed_results
    )

    grid_summary.to_csv(
        output_root / "transaction_cost_grid_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Ranking tables
    # ---------------------------------------------------------
    all_costs = [0] + list(args.cost_bps)

    for cost in all_costs:
        subset = grid_summary[
            grid_summary["cost_bps"] == cost
        ].copy()

        if subset.empty:
            continue

        subset.sort_values(
            "sharpe_mean",
            ascending=False,
        ).to_csv(
            output_root / f"top_by_sharpe_{int(cost)}bp.csv",
            index=False,
        )

        subset.sort_values(
            "cagr_mean",
            ascending=False,
        ).to_csv(
            output_root / f"top_by_cagr_{int(cost)}bp.csv",
            index=False,
        )

        subset.sort_values(
            "mdd_mean",
            ascending=False,
        ).to_csv(
            output_root / f"top_by_mdd_{int(cost)}bp.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Best Sharpe configuration at each transaction-cost level
    # ---------------------------------------------------------
    best_rows = []

    for cost in all_costs:
        subset = grid_summary[
            grid_summary["cost_bps"] == cost
        ]

        if subset.empty:
            continue

        best = subset.loc[
            subset["sharpe_mean"].idxmax()
        ].copy()

        best_rows.append(best)

    if best_rows:
        pd.DataFrame(best_rows).to_csv(
            output_root / "best_sharpe_by_cost.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Failed files
    # ---------------------------------------------------------
    if failed:
        pd.DataFrame(failed).to_csv(
            output_root / "failed_files.csv",
            index=False,
        )

    print()
    print("Completed.")
    print(
        f"Seed results : "
        f"{output_root / 'transaction_cost_seed_results.csv'}"
    )
    print(
        f"Grid summary : "
        f"{output_root / 'transaction_cost_grid_summary.csv'}"
    )

    if failed:
        print(
            f"WARNING: {len(failed)} files failed. "
            f"See failed_files.csv"
        )


if __name__ == "__main__":
    main()