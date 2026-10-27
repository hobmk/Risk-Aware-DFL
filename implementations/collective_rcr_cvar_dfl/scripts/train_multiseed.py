from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.collective_rcr_cvar_dfl.src.core import (
    CollectiveCVaRDataset,
    build_collective_cvar_layer,
    collective_cvar_cost,
)
from implementations.dfl_mvo_lee2025.src.dataset import (
    FeatureStandardizedSubset,
    chronological_split,
    fit_feature_standardizer,
)
from implementations.dfl_mvo_lee2025.src.losses import combined_loss
from implementations.dfl_mvo_lee2025.src.metrics import calculate_portfolio_metrics
from implementations.dfl_mvo_lee2025.src.model import ReturnMLP
from implementations.dfl_mvo_lee2025.src.optimization import covariance_to_risk_factor


SOLVER_REG = 1e-10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mse-scale", type=float, default=30.0)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--cvar-alpha", type=float, default=0.90)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument(
        "--output-dir",
        default="implementations/collective_rcr_cvar_dfl/outputs",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class IndexedCollectiveCVaRDataset(CollectiveCVaRDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["sample_index"] = int(index)
        return sample


class CollectiveRCRCVaRDFL(nn.Module):
    def __init__(self, n_assets, lookback, hidden_dim, dropout, risk_aversion,
                 max_weight, gamma, cvar_alpha):
        super().__init__()

        self.return_model = ReturnMLP(
            n_assets=n_assets,
            lookback=lookback,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.cvar_layer = build_collective_cvar_layer(
            n_assets=n_assets,
            n_scenarios=lookback,
            gamma=gamma,
            cvar_alpha=cvar_alpha,
            max_weight=max_weight,
        )

        self.risk_aversion = risk_aversion
        self.gamma = gamma
        self.cvar_alpha = cvar_alpha

    def forward(self, features, covariance, scenarios, probabilities):
        predicted_returns = self.return_model(features)

        mu = predicted_returns.to(device="cpu", dtype=torch.float64)
        covariance = covariance.to(device="cpu", dtype=torch.float64)
        scenarios = scenarios.to(device="cpu", dtype=torch.float64)
        probabilities = probabilities.to(device="cpu", dtype=torch.float64)

        risk_factor = covariance_to_risk_factor(
            covariance,
            risk_aversion=self.risk_aversion,
            jitter=0.0,
        )

        weights, tau, u = self.cvar_layer(
            mu,
            risk_factor,
            scenarios,
            probabilities,
        )

        return predicted_returns, weights, tau, u, risk_factor


def decision_cost(weights, tau, u, true_returns, risk_factor,
                  probabilities, gamma, cvar_alpha):
    cost = collective_cvar_cost(
        weights=weights,
        tau=tau,
        u=u,
        true_returns=true_returns,
        risk_factor=risk_factor,
        probabilities=probabilities,
        gamma=gamma,
        cvar_alpha=cvar_alpha,
    )

    regularizer = SOLVER_REG * (
        u.square().sum(dim=-1)
        + tau.square()
    )

    return cost + regularizer


def precompute_oracle_costs(dataset, layer, args):
    print("Precomputing oracle decision costs...")

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    oracle_cost = torch.empty(
        len(dataset),
        dtype=torch.float64,
    )

    for batch_index, batch in enumerate(loader):
        indices = batch["sample_index"].long()

        targets = batch["target"].to(torch.float64)
        covariance = batch["covariance"].to(torch.float64)
        scenarios = batch["residual_scenarios"].to(torch.float64)
        probabilities = batch["scenario_probabilities"].to(torch.float64)

        risk_factor = covariance_to_risk_factor(
            covariance,
            risk_aversion=args.risk_aversion,
            jitter=0.0,
        )

        weights, tau, u = layer(
            targets,
            risk_factor,
            scenarios,
            probabilities,
        )

        costs = decision_cost(
            weights=weights,
            tau=tau,
            u=u,
            true_returns=targets,
            risk_factor=risk_factor,
            probabilities=probabilities,
            gamma=args.gamma,
            cvar_alpha=args.cvar_alpha,
        )

        oracle_cost[indices] = costs.detach()

        if batch_index % 20 == 0:
            print(
                f"  oracle batch "
                f"{batch_index + 1}/{len(loader)}"
            )

    print(
        f"Oracle cache complete | "
        f"mean={oracle_cost.mean().item():.8f}"
    )

    return oracle_cost


def run_epoch(model, loader, device, oracle_costs, args, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_regret = 0.0
    total_mse = 0.0
    total_samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)
        covariance = batch["covariance"]
        scenarios = batch["residual_scenarios"]
        probabilities = batch["scenario_probabilities"]
        sample_indices = batch["sample_index"].long()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            (
                predicted_returns,
                weights,
                tau,
                u,
                risk_factor,
            ) = model(
                features,
                covariance,
                scenarios,
                probabilities,
            )

            true_returns = targets.to(
                device="cpu",
                dtype=torch.float64,
            )

            probabilities_cpu = probabilities.to(
                device="cpu",
                dtype=torch.float64,
            )

            predicted_cost = decision_cost(
                weights=weights,
                tau=tau,
                u=u,
                true_returns=true_returns,
                risk_factor=risk_factor,
                probabilities=probabilities_cpu,
                gamma=args.gamma,
                cvar_alpha=args.cvar_alpha,
            )

            oracle = oracle_costs[
                sample_indices
            ].to(
                device=predicted_cost.device,
                dtype=predicted_cost.dtype,
            )

            regret = (
                predicted_cost - oracle
            ).mean()

            mse = nn.functional.mse_loss(
                predicted_returns,
                targets,
            )

            loss = combined_loss(
                regret_loss=regret,
                mse_loss=mse,
                alpha=args.alpha,
                mse_scale=args.mse_scale,
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)

        total_loss += loss.detach().item() * batch_size
        total_regret += regret.detach().item() * batch_size
        total_mse += mse.detach().item() * batch_size
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_regret / total_samples,
        total_mse / total_samples,
    )


def drift_turnover(weights, returns):
    turnover = np.zeros(len(weights), dtype=np.float64)

    for t in range(1, len(weights)):
        prev_w = weights[t - 1]
        prev_r = returns[t - 1]

        denominator = (
            1.0 + np.dot(prev_w, prev_r)
        )

        if abs(denominator) <= 1e-15:
            drifted = prev_w
        else:
            drifted = (
                prev_w
                * (1.0 + prev_r)
                / denominator
            )

        turnover[t] = (
            0.5
            * np.abs(
                weights[t] - drifted
            ).sum()
        )

    return turnover


def portfolio_metrics(returns):
    returns = torch.tensor(
        returns,
        dtype=torch.float64,
    )

    metrics, _, _ = calculate_portfolio_metrics(
        returns
    )

    return metrics


def evaluate_and_save(model, loader, device, tickers, output_dir, args):
    model.eval()

    dates = []
    predicted_returns_all = []
    weights_all = []
    returns_all = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)

            (
                predicted_returns,
                weights,
                _,
                _,
                _,
            ) = model(
                features,
                batch["covariance"],
                batch["residual_scenarios"],
                batch["scenario_probabilities"],
            )

            dates.extend(
                list(batch["target_date"])
            )

            predicted_returns_all.append(
                predicted_returns.detach()
                .cpu()
                .numpy()
            )

            weights_all.append(
                weights.detach()
                .cpu()
                .numpy()
            )

            returns_all.append(
                batch["target"]
                .detach()
                .cpu()
                .numpy()
            )

    predicted_returns = np.concatenate(
        predicted_returns_all
    )

    weights = np.concatenate(
        weights_all
    )

    asset_returns = np.concatenate(
        returns_all
    )

    gross_returns = np.sum(
        weights * asset_returns,
        axis=1,
    )

    turnover = drift_turnover(
        weights,
        asset_returns,
    )

    returns_5bp = (
        gross_returns
        - 0.0005 * turnover
    )

    returns_10bp = (
        gross_returns
        - 0.0010 * turnover
    )

    gross = portfolio_metrics(
        gross_returns
    )

    cost5 = portfolio_metrics(
        returns_5bp
    )

    cost10 = portfolio_metrics(
        returns_10bp
    )

    asset_rows = []

    for t, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            asset_rows.append({
                "date": date,
                "ticker": ticker,
                "predicted_return":
                    predicted_returns[t, j],
                "predicted_weight":
                    weights[t, j],
                "true_return":
                    asset_returns[t, j],
            })

    pd.DataFrame(asset_rows).to_csv(
        output_dir / "asset_predictions.csv",
        index=False,
    )

    pd.DataFrame({
        "date": dates,
        "portfolio_return": gross_returns,
        "turnover": turnover,
        "return_5bp": returns_5bp,
        "return_10bp": returns_10bp,
    }).to_csv(
        output_dir / "test_daily_portfolio.csv",
        index=False,
    )

    return {
        "annual_return": gross[
            "annualized_return_cagr"
        ],
        "volatility": gross[
            "annualized_volatility"
        ],
        "sharpe": gross[
            "sharpe_ratio"
        ],
        "mdd": gross[
            "maximum_drawdown"
        ],
        "annualized_turnover":
            float(turnover.mean() * 252.0),
        "annual_return_5bp":
            cost5["annualized_return_cagr"],
        "sharpe_5bp":
            cost5["sharpe_ratio"],
        "annual_return_10bp":
            cost10["annualized_return_cagr"],
        "sharpe_10bp":
            cost10["sharpe_ratio"],
    }


def train_seed(
    seed,
    dataset,
    train_set,
    validation_set,
    test_set,
    feature_mean,
    feature_std,
    oracle_costs,
    args,
):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = CollectiveRCRCVaRDFL(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        gamma=args.gamma,
        cvar_alpha=args.cvar_alpha,
    )

    model.return_model = (
        model.return_model
        .float()
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = (
        PROJECT_ROOT
        / args.output_dir
        / f"gamma_{args.gamma:.2f}"
        / f"seed_{seed}"
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. "
                f"Use --overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_validation = float("inf")
    best_epoch = 0
    patience_count = 0

    checkpoint_path = (
        output_dir / "best_model.pt"
    )

    print()
    print("=" * 90)
    print(
        f"SEED {seed} | "
        f"Collective RCR-CVaR DFL"
    )
    print("=" * 90)

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss, train_regret, train_mse = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            oracle_costs=oracle_costs,
            args=args,
            optimizer=optimizer,
        )

        val_loss, val_regret, val_mse = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            oracle_costs=oracle_costs,
            args=args,
            optimizer=None,
        )

        history.append({
            "epoch": epoch,
            "train_combined": train_loss,
            "train_regret": train_regret,
            "train_mse": train_mse,
            "validation_combined": val_loss,
            "validation_regret": val_regret,
            "validation_mse": val_mse,
        })

        pd.DataFrame(
            history
        ).to_csv(
            output_dir / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train={train_loss:.8f} | "
            f"Regret={train_regret:.8f} | "
            f"MSE={train_mse:.8f} | "
            f"Val={val_loss:.8f}"
        )

        if val_loss < best_validation:
            best_validation = val_loss
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "model_state_dict":
                        model.return_model.state_dict(),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_combined":
                        best_validation,
                },
                checkpoint_path,
            )

        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(
                f"Early stopping | "
                f"epoch={epoch}"
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.return_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_loss, test_regret, test_mse = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        oracle_costs=oracle_costs,
        args=args,
        optimizer=None,
    )

    metrics = evaluate_and_save(
        model=model,
        loader=test_loader,
        device=device,
        tickers=dataset.tickers,
        output_dir=output_dir,
        args=args,
    )

    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_combined":
            best_validation,
        "test_combined": test_loss,
        "test_regret": test_regret,
        "test_mse": test_mse,
        **metrics,
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"BEST epoch={best_epoch} | "
        f"Return={metrics['annual_return']:.4f} | "
        f"Vol={metrics['volatility']:.4f} | "
        f"Sharpe={metrics['sharpe']:.4f} | "
        f"MDD={metrics['mdd']:.4f} | "
        f"S5={metrics['sharpe_5bp']:.4f} | "
        f"S10={metrics['sharpe_10bp']:.4f}"
    )

    return summary


def main():
    args = parse_args()

    price_csv = Path(
        args.price_csv
    )

    if not price_csv.is_absolute():
        price_csv = (
            PROJECT_ROOT
            / price_csv
        )

    dataset = IndexedCollectiveCVaRDataset(
        price_csv=price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float32,
        tail_q=args.tail_q,
    )

    train_raw, validation_raw, test_raw = (
        chronological_split(
            dataset=dataset,
            train_end=args.train_end,
            validation_end=args.validation_end,
        )
    )

    feature_mean, feature_std = (
        fit_feature_standardizer(
            dataset=dataset,
            train_subset=train_raw,
        )
    )

    train_set = FeatureStandardizedSubset(
        train_raw,
        feature_mean,
        feature_std,
    )

    validation_set = FeatureStandardizedSubset(
        validation_raw,
        feature_mean,
        feature_std,
    )

    test_set = FeatureStandardizedSubset(
        test_raw,
        feature_mean,
        feature_std,
    )

    oracle_layer = build_collective_cvar_layer(
        n_assets=dataset.n_assets,
        n_scenarios=args.lookback,
        gamma=args.gamma,
        cvar_alpha=args.cvar_alpha,
        max_weight=args.max_weight,
    )

    oracle_costs = precompute_oracle_costs(
        dataset=dataset,
        layer=oracle_layer,
        args=args,
    )

    print()
    print(
        f"Train      : "
        f"{dataset.target_dates[train_raw.indices[0]].date()} "
        f"~ "
        f"{dataset.target_dates[train_raw.indices[-1]].date()}"
    )
    print(
        f"Validation : "
        f"{dataset.target_dates[validation_raw.indices[0]].date()} "
        f"~ "
        f"{dataset.target_dates[validation_raw.indices[-1]].date()}"
    )
    print(
        f"Test       : "
        f"{dataset.target_dates[test_raw.indices[0]].date()} "
        f"~ "
        f"{dataset.target_dates[test_raw.indices[-1]].date()}"
    )

    summaries = []

    for seed in args.seeds:
        summaries.append(
            train_seed(
                seed=seed,
                dataset=dataset,
                train_set=train_set,
                validation_set=validation_set,
                test_set=test_set,
                feature_mean=feature_mean,
                feature_std=feature_std,
                oracle_costs=oracle_costs,
                args=args,
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    root = (
        PROJECT_ROOT
        / args.output_dir
        / f"gamma_{args.gamma:.2f}"
    )

    summary_df.to_csv(
        root / "multiseed_summary.csv",
        index=False,
    )

    numeric = [
        "annual_return",
        "volatility",
        "sharpe",
        "mdd",
        "annualized_turnover",
        "sharpe_5bp",
        "sharpe_10bp",
        "test_mse",
        "test_regret",
    ]

    stats = pd.DataFrame({
        "metric": numeric,
        "mean": [
            summary_df[x].mean()
            for x in numeric
        ],
        "std": [
            summary_df[x].std(ddof=1)
            for x in numeric
        ],
    })

    stats.to_csv(
        root / "multiseed_stats.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("MULTI-SEED SUMMARY")
    print("=" * 90)
    print(
        stats.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        summary_df[
            [
                "seed",
                "best_epoch",
                "annual_return",
                "volatility",
                "sharpe",
                "mdd",
                "sharpe_5bp",
                "sharpe_10bp",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()