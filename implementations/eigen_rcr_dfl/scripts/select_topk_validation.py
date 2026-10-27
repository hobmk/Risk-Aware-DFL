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
from implementations.eigen_rcr_dfl.src.dataset import (
    EigenRCRDataset,
    chronological_split,
)
from implementations.eigen_rcr_dfl.src.model import EigenRCRDFL


DEFAULT_KS = {1, 3, 5, 10, 30}
DEFAULT_ETAS = {0.10, 0.25, 0.50, 0.75, 1.00}
DEFAULT_SCALES = {"none", "preserve_residual_trace"}
DEFAULT_SEEDS = {42, 43, 44}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select Top-K configuration using validation performance only."
    )
    parser.add_argument(
        "--root",
        default="implementations/eigen_rcr_dfl/outputs/k_eta_grid_standardized",
    )
    parser.add_argument(
        "--output-dir",
        default="implementations/eigen_rcr_dfl/outputs/validation_selection/topk",
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


def same_float(a, b, tol=1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_target_run(config: dict, risk_aversion: float) -> bool:
    k = int(config.get("eigen_k", -1))
    eta = float(config.get("eta", -1.0))
    scale = config.get("scale_mode")
    seed = int(config.get("seed", -1))

    return (
        k in DEFAULT_KS
        and any(same_float(eta, x) for x in DEFAULT_ETAS)
        and scale in DEFAULT_SCALES
        and seed in DEFAULT_SEEDS
        and same_float(config.get("risk_aversion", -1.0), risk_aversion)
        and bool(config.get("standardize_inputs", True))
    )


def dataset_key(config: dict) -> tuple:
    return (
        str(config["price_csv"]),
        int(config["lookback"]),
        str(config["date_column"]),
        str(config["return_type"]),
        float(config["covariance_jitter"]),
        str(config["market_mode"]),
        str(config.get("market_price_csv")),
        str(config.get("market_column")),
        float(config["risk_free_rate"]),
        bool(not config["no_capm_intercept"]),
        int(config["eigen_k"]),
        str(config["scale_mode"]),
        str(config["train_end"]),
        str(config["validation_end"]),
    )


def build_validation_dataset(config: dict):
    dataset = EigenRCRDataset(
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
        eigen_k=int(config["eigen_k"]),
        scale_mode=config["scale_mode"],
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
        state = torch.load(path, map_location="cpu", weights_only=False)
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
    dataset: EigenRCRDataset,
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
        eta=float(config["eta"]),
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

    metrics = evaluate_common_portfolio(
        portfolio_returns=portfolio_returns,
        weights=weights,
        asset_returns=targets,
        cost_bps=(0.0, 5.0, 10.0),
        active_threshold=1e-3,
    )

    return metrics


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

        if is_target_run(config, args.risk_aversion):
            run_dir = config_path.parent

            if not (run_dir / "best_model.pt").exists():
                continue

            selected_paths.append(config_path)

    print(f"Candidate runs found: {len(selected_paths)}")

    if not selected_paths:
        raise RuntimeError(
            f"조건에 맞는 기존 run이 없습니다: {root}"
        )

    dataset_cache = {}
    rows = []

    for index, config_path in enumerate(selected_paths, start=1):
        run_dir = config_path.parent
        config = load_json(config_path)

        key = dataset_key(config)

        if key not in dataset_cache:
            dataset_cache[key] = build_validation_dataset(config)

        dataset, validation_dataset = dataset_cache[key]

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
            "seed": int(config["seed"]),
            "k": int(config["eigen_k"]),
            "eta": float(config["eta"]),
            "scale_mode": config["scale_mode"],
            "lambda": float(config["risk_aversion"]),
        }

        row.update(metrics)
        rows.append(row)

        print(
            f"[{index:03d}/{len(selected_paths):03d}] "
            f"K={row['k']:2d} | "
            f"eta={row['eta']:.2f} | "
            f"scale={row['scale_mode']} | "
            f"seed={row['seed']} | "
            f"Val Sharpe={row['gross_sharpe']:.4f}"
        )

    runs = pd.DataFrame(rows)

    runs.to_csv(
        output_dir / "validation_runs.csv",
        index=False,
    )

    group_cols = ["k", "eta", "scale_mode"]

    summary = (
        runs.groupby(group_cols, as_index=False)
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
        "selection_rule": "highest mean validation gross Sharpe across seeds 42-44",
        "validation_only": True,
        "selection_lambda": float(args.risk_aversion),
        "reuse_for_final_lambdas": [1.0, 25.0],
        "seeds": sorted(DEFAULT_SEEDS),
        "selected": {
            "k": int(best["k"]),
            "eta": float(best["eta"]),
            "scale_mode": str(best["scale_mode"]),
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
        output_dir / "selected_topk_config.json",
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
    print("TOP-K VALIDATION SELECTION")
    print("=" * 100)
    print(
        summary.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSelected:")
    print(json.dumps(selected, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()