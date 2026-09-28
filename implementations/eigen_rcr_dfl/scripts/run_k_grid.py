from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 30])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--scale-mode",
        choices=[
            "none",
            "preserve_residual_trace",
            "match_base_trace",
        ],
        default="none",
    )

    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    # 지정하면 기존 완료 결과도 다시 실행
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def output_directory(
    root: Path,
    scale_mode: str,
    eta: float,
    alpha: float,
    k: int,
    seed: int,
) -> Path:
    return (
        root
        / f"scale_{scale_mode}"
        / f"eta_{eta:.2f}"
        / f"alpha_{alpha:.2f}"
        / f"k_{k}"
        / f"seed_{seed}"
    )


def summary_path(
    root: Path,
    scale_mode: str,
    eta: float,
    alpha: float,
    seed: int,
) -> Path:
    return root / (
        f"summary_{scale_mode}"
        f"_eta_{eta:.2f}"
        f"_alpha_{alpha:.2f}"
        f"_seed_{seed}.csv"
    )


def result_files_complete(output_dir: Path) -> bool:
    required_files = [
        "config.json",
        "best_model.pt",
        "history.csv",
        "test_metrics.csv",
        "portfolio_summary.csv",
    ]

    return all(
        (output_dir / filename).exists()
        for filename in required_files
    )


def config_matches(
    output_dir: Path,
    k: int,
    seed: int,
    scale_mode: str,
    eta: float,
    alpha: float,
    epochs: int,
    patience: int,
) -> bool:
    config_path = output_dir / "config.json"

    if not config_path.exists():
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    expected_exact = {
        "eigen_k": k,
        "seed": seed,
        "scale_mode": scale_mode,
        "epochs": epochs,
        "patience": patience,
    }

    for key, expected in expected_exact.items():
        if config.get(key) != expected:
            return False

    expected_float = {
        "eta": eta,
        "alpha": alpha,
    }

    for key, expected in expected_float.items():
        value = config.get(key)

        if value is None:
            return False

        if not math.isclose(
            float(value),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return False

    return True


def load_result(
    output_dir: Path,
    k: int,
    seed: int,
    scale_mode: str,
    eta: float,
    alpha: float,
) -> dict:
    portfolio = pd.read_csv(
        output_dir / "portfolio_summary.csv"
    ).iloc[0].to_dict()

    test = pd.read_csv(
        output_dir / "test_metrics.csv"
    ).iloc[0].to_dict()

    config_path = output_dir / "config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    row = {
        "k": k,
        "seed": seed,
        "scale_mode": scale_mode,
        "eta": eta,
        "alpha": alpha,
        "mean_retained_ratio": config.get(
            "mean_retained_ratio",
            float("nan"),
        ),
    }

    row.update({
        f"portfolio_{key}": value
        for key, value in portfolio.items()
    })

    row.update({
        f"test_{key}": value
        for key, value in test.items()
    })

    return row


def save_summary(
    rows: list[dict],
    path: Path,
) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)

    if path.exists():
        old_df = pd.read_csv(path)

        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )
    else:
        combined = new_df

    combined = combined.drop_duplicates(
        subset=[
            "scale_mode",
            "k",
            "eta",
            "alpha",
            "seed",
        ],
        keep="last",
    )

    combined = combined.sort_values(
        ["k"]
    ).reset_index(drop=True)

    combined.to_csv(
        path,
        index=False,
    )

    return combined


def main():
    args = parse_args()

    root = Path(
        "implementations/eigen_rcr_dfl/outputs/k_grid"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_rows = []

    skipped = 0
    executed = 0

    for k in args.ks:
        output_dir = output_directory(
            root=root,
            scale_mode=args.scale_mode,
            eta=args.eta,
            alpha=args.alpha,
            k=k,
            seed=args.seed,
        )

        complete = result_files_complete(
            output_dir
        )

        matches = (
            complete
            and config_matches(
                output_dir=output_dir,
                k=k,
                seed=args.seed,
                scale_mode=args.scale_mode,
                eta=args.eta,
                alpha=args.alpha,
                epochs=args.epochs,
                patience=args.patience,
            )
        )

        print("\n" + "=" * 90)
        print(
            f"K={k} | "
            f"scale={args.scale_mode} | "
            f"eta={args.eta:.2f} | "
            f"alpha={args.alpha:.2f} | "
            f"seed={args.seed}"
        )
        print("=" * 90)

        if complete and matches and not args.overwrite:
            print(
                f"[SKIP] Completed result already exists:\n"
                f"       {output_dir}"
            )

            skipped += 1

        else:
            if output_dir.exists():
                if complete and not matches:
                    print(
                        "[RERUN] Existing result config does not "
                        "match requested settings."
                    )

                elif not complete:
                    print(
                        "[RERUN] Existing output is incomplete."
                    )

            command = [
                sys.executable,
                "-m",
                "implementations.eigen_rcr_dfl.scripts.train_eigen_rcr",

                "--eigen-k",
                str(k),

                "--scale-mode",
                args.scale_mode,

                "--eta",
                str(args.eta),

                "--alpha",
                str(args.alpha),

                "--epochs",
                str(args.epochs),

                "--patience",
                str(args.patience),

                "--seed",
                str(args.seed),

                "--output-dir",
                str(output_dir),
            ]

            # 기존 폴더가 있거나 overwrite를 명시했으면
            # train script에 overwrite 전달
            if output_dir.exists() or args.overwrite:
                command.append("--overwrite")

            subprocess.run(
                command,
                check=True,
            )

            executed += 1

        row = load_result(
            output_dir=output_dir,
            k=k,
            seed=args.seed,
            scale_mode=args.scale_mode,
            eta=args.eta,
            alpha=args.alpha,
        )

        current_rows.append(row)

        path = summary_path(
            root=root,
            scale_mode=args.scale_mode,
            eta=args.eta,
            alpha=args.alpha,
            seed=args.seed,
        )

        save_summary(
            current_rows,
            path,
        )

    path = summary_path(
        root=root,
        scale_mode=args.scale_mode,
        eta=args.eta,
        alpha=args.alpha,
        seed=args.seed,
    )

    summary = pd.read_csv(path)

    requested = summary[
        summary["k"].isin(args.ks)
    ].copy()

    print("\n" + "=" * 120)
    print("K GRID SUMMARY")
    print("=" * 120)

    columns = [
        "k",
        "mean_retained_ratio",
        "portfolio_cumulative_return",
        "portfolio_annualized_return",
        "portfolio_volatility",
        "portfolio_sharpe",
        "portfolio_max_drawdown",
        "portfolio_annualized_turnover",
        "portfolio_active_assets",
        "test_mse",
        "test_regret",
    ]

    available_columns = [
        column
        for column in columns
        if column in requested.columns
    ]

    print(
        requested[
            available_columns
        ].sort_values("k").to_string(
            index=False
        )
    )

    print("\n" + "-" * 60)
    print(f"Executed : {executed}")
    print(f"Skipped  : {skipped}")
    print(f"Requested: {len(args.ks)}")
    print(f"Summary  : {path}")
    print("-" * 60)


if __name__ == "__main__":
    main()