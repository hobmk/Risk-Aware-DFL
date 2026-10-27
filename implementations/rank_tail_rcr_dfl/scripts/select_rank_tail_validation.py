from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from implementations.common.portfolio_evaluation import (
    evaluate_portfolio as evaluate_common_portfolio,
)
from implementations.dfl_mvo_lee2025.src.dataset import FeatureStandardizedSubset
from implementations.eigen_rcr_dfl.src.dataset import chronological_split
from implementations.eigen_rcr_dfl.src.model import EigenRCRDFL
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


Q_VALUES = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}
SEEDS = {42, 43, 44, 45, 46}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select Rank-Tail q using validation performance only."
    )
    parser.add_argument(
        "--root",
        default="implementations/rank_tail_rcr_dfl/outputs/standardized_sensitivity",
    )
    parser.add_argument(
        "--output-dir",
        default="implementations/rank_tail_rcr_dfl/outputs/validation_selection",
    )
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def same_float(a, b, tol=1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def is_target_run(config: dict, risk_aversion: float) -> bool:
    q = float(config.get("tail_q", -1.0))
    seed = int(config.get("seed", -1))

    return (
        any(same_float(q, x) for x in Q_VALUES)
        and seed in SEEDS
        and same_float(config.get("risk_aversion", -1.0), risk_aversion)
        and same_float(config.get("alpha", -1.0), 0.5)
        and config.get("tail_side") == "down"
        and bool(config.get("standardize_inputs", False))
    )


def build_validation_dataset(config: dict):
    dataset = RankTailRCRDataset(
        price_csv=config["price_csv"],
        lookback=int(config["lookback"]),
        date_column=config["date_column"],
        return_type=config["return_type"],
        covariance_jitter=float(config["covariance_jitter"]),
        market_mode=config["market_mode"],
        market_price_csv=config.get("market_price_csv"),
        market_column=config.get("market_column"),
        risk_free_rate=float(config["risk_free_rate"]),
        fit_intercept=not bool(config["no_capm_intercept"]),
        tail_q=float(config["tail_q"]),
        tail_side=config["tail_side"],
        risk_cache_dir=config.get("risk_cache_dir"),
        dtype=torch.float64,
    )

    _, validation_dataset, _ = chronological_split(
        dataset,
        config["train_end"],
        config["validation_end"],
    )

    return dataset, validation_dataset


def load_standardizer(run_dir: Path, config: dict):
    path = run_dir / "feature_standardizer.pt"

    if path.exists():
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        return state["feature_mean"], state["feature_std"]

    if "feature_mean" not in config or "feature_std" not in config:
        raise FileNotFoundError(
            f"표준화 정보를 찾을 수 없습니다: {run_dir}"
        )

    return (
        torch.tensor(config["feature_mean"], dtype=torch.float64),
        torch.tensor(config["feature_std"], dtype=torch.float64),
    )


def evaluate_run(
    run_dir: Path,
    config: dict,
    dataset,
    validation_dataset,
    device: torch.device,
    batch_size: int,
) -> dict:
    feature_mean, feature_std = load_standardizer(run_dir, config)

    validation_dataset = FeatureStandardizedSubset(
        validation_dataset,
        feature_mean,
        feature_std,
    )

    loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = EigenRCRDFL(
        n_assets=dataset.n_assets,
        lookback=int(config["lookback"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        risk_aversion=float(config["risk_aversion"]),
        max_weight=float(config["max_weight"]),
        eta=0.0,
        effective_jitter=float(config["effective_jitter"]),
    ).to(device)

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_weights = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(
                device=device,
                dtype=torch.float32,
            )

            covariance = batch["covariance"].to(
                device="cpu",
                dtype=torch.float64,
            )

            eigen_risk = batch["eigen_risk"].to(
                device="cpu",
                dtype=torch.float64,
            )

            output = model(
                features=features,
                covariance=covariance,
                eigen_risk=eigen_risk,
            )

            all_weights.append(
                output.predicted_weights.detach().cpu().numpy()
            )

            all_targets.append(
                batch["target"].detach().cpu().numpy()
            )

    weights = np.concatenate(all_weights, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    portfolio_returns = np.sum(
        weights * targets,
        axis=1,
    )

    return evaluate_common_portfolio(
        portfolio_returns=portfolio_returns,
        weights=weights,
        asset_returns=targets,
        cost_bps=(0.0, 5.0, 10.0),
        active_threshold=1e-3,
    )


def main():
    args = parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    config_paths = sorted(root.rglob("config.json"))
    selected_paths = []

    for config_path in config_paths:
        config = load_json(config_path)

        if (
            is_target_run(config, args.risk_aversion)
            and (config_path.parent / "best_model.pt").exists()
            and (config_path.parent / "feature_standardizer.pt").exists()
        ):
            selected_paths.append(config_path)

    print(f"Candidate runs found: {len(selected_paths)}")
    print(f"Expected runs       : {len(Q_VALUES) * len(SEEDS)}")

    if not selected_paths:
        raise RuntimeError(
            "조건에 맞는 standardized Rank-Tail run이 없습니다."
        )

    rows = []

    for index, config_path in enumerate(selected_paths, start=1):
        run_dir = config_path.parent
        config = load_json(config_path)

        dataset, validation_dataset = build_validation_dataset(config)

        metrics = evaluate_run(
            run_dir=run_dir,
            config=config,
            dataset=dataset,
            validation_dataset=validation_dataset,
            device=device,
            batch_size=args.batch_size,
        )

        row = {
            "run_dir": str(run_dir),
            "q": float(config["tail_q"]),
            "seed": int(config["seed"]),
            "lambda": float(config["risk_aversion"]),
        }

        row.update(metrics)
        rows.append(row)

        print(
            f"[{index:02d}/{len(selected_paths):02d}] "
            f"q={row['q']:.2f} | "
            f"seed={row['seed']} | "
            f"Val Sharpe={row['gross_sharpe']:.4f}"
        )

    runs = pd.DataFrame(rows)

    runs.to_csv(
        output_dir / "validation_runs.csv",
        index=False,
    )

    summary = (
        runs.groupby("q", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            val_return_mean=("gross_annualized_return", "mean"),
            val_return_std=("gross_annualized_return", "std"),
            val_vol_mean=("gross_volatility", "mean"),
            val_vol_std=("gross_volatility", "std"),
            val_sharpe_mean=("gross_sharpe", "mean"),
            val_sharpe_std=("gross_sharpe", "std"),
            val_sortino_mean=("gross_sortino", "mean"),
            val_sortino_std=("gross_sortino", "std"),
            val_mdd_mean=("gross_max_drawdown", "mean"),
            val_mdd_std=("gross_max_drawdown", "std"),
            turnover_mean=("annualized_turnover", "mean"),
            hhi_mean=("hhi", "mean"),
            effective_n_mean=("effective_n", "mean"),
        )
    )

    summary = summary.sort_values(
        ["val_sharpe_mean", "turnover_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)

    summary.to_csv(
        output_dir / "validation_summary.csv",
        index=False,
    )

    best = summary.iloc[0]

    selected = {
        "selection_rule": (
            "highest mean validation gross Sharpe across seeds 42-46"
        ),
        "validation_only": True,
        "selection_lambda": float(args.risk_aversion),
        "reuse_for_final_lambdas": [1.0, 25.0],
        "seeds": sorted(SEEDS),
        "candidate_q_values": sorted(Q_VALUES),
        "selected": {
            "q": float(best["q"]),
            "tail_side": "down",
            "validation_sharpe_mean": float(best["val_sharpe_mean"]),
            "validation_sharpe_std": float(best["val_sharpe_std"]),
            "validation_return_mean": float(best["val_return_mean"]),
            "validation_volatility_mean": float(best["val_vol_mean"]),
            "validation_turnover_mean": float(best["turnover_mean"]),
            "validation_hhi_mean": float(best["hhi_mean"]),
            "validation_effective_n_mean": float(best["effective_n_mean"]),
        },
    }

    with open(
        output_dir / "selected_rank_tail_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            selected,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("RANK-TAIL VALIDATION SELECTION")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSelected:")
    print(
        json.dumps(
            selected,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()