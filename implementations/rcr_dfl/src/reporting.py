from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .decision_model import RCRMLPWithMarkowitz
from .losses import markowitz_cost


def calculate_portfolio_metrics(
    portfolio_returns: torch.Tensor,
    periods_per_year: int = 252,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """기존 DFL-MVO 재현 코드와 동일한 포트폴리오 성과 정의를 사용한다."""

    returns = (
        portfolio_returns.detach()
        .cpu()
        .to(torch.float64)
        .flatten()
    )

    if returns.numel() == 0:
        raise ValueError("portfolio_returns가 비어 있습니다.")

    wealth = torch.cumprod(1.0 + returns, dim=0)
    running_max = torch.cummax(wealth, dim=0).values
    drawdown = 1.0 - wealth / running_max

    final_wealth = wealth[-1].item()
    n_periods = int(returns.numel())

    annualized_return_cagr = (
        final_wealth ** (periods_per_year / n_periods) - 1.0
        if final_wealth > 0
        else float("nan")
    )

    mean_return = returns.mean().item()
    return_std = (
        returns.std(unbiased=True).item()
        if n_periods > 1
        else 0.0
    )

    annualized_return_mean = periods_per_year * mean_return
    annualized_volatility = math.sqrt(periods_per_year) * return_std
    sharpe_ratio = (
        math.sqrt(periods_per_year) * mean_return / return_std
        if return_std > 0
        else float("nan")
    )

    metrics = {
        "total_return": final_wealth - 1.0,
        "final_wealth": final_wealth,
        "annualized_return_cagr": annualized_return_cagr,
        "annualized_return_mean": annualized_return_mean,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": drawdown.max().item(),
    }

    return metrics, wealth, drawdown


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )


def evaluate_test_portfolio(
    model: RCRMLPWithMarkowitz,
    test_loader: DataLoader,
    tickers: list[str],
    device: torch.device,
    active_threshold: float,
    cap_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()

    daily_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []

    portfolio_returns: list[float] = []
    equal_weight_returns: list[float] = []

    previous_weight: torch.Tensor | None = None
    max_weight = float(model.max_weight)

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(
                device=device,
                dtype=torch.float32,
            )
            true_returns = batch["target"].to(
                device="cpu",
                dtype=torch.float64,
            )
            covariance = batch["covariance"].to(
                device="cpu",
                dtype=torch.float64,
            )
            a_res = batch["a_res"].to(
                device="cpu",
                dtype=torch.float64,
            )
            dates = list(batch["target_date"])

            output = model(
                features=features,
                covariance=covariance,
                a_res=a_res,
            )

            predicted_returns = (
                output.predicted_returns.detach()
                .to(device="cpu", dtype=torch.float64)
            )
            predicted_weights = (
                output.predicted_weights.detach()
                .to(device="cpu", dtype=torch.float64)
            )
            effective_covariance = (
                output.effective_covariance.detach()
                .to(device="cpu", dtype=torch.float64)
            )
            risk_factor = (
                output.risk_factor.detach()
                .to(device="cpu", dtype=torch.float64)
            )

            oracle_weights = model.solve_oracle(
                true_returns=true_returns,
                risk_factor=risk_factor,
            ).to(device="cpu", dtype=torch.float64)

            predicted_cost = markowitz_cost(
                weights=predicted_weights,
                true_returns=true_returns,
                risk_factor=risk_factor,
            )
            oracle_cost = markowitz_cost(
                weights=oracle_weights,
                true_returns=true_returns,
                risk_factor=risk_factor,
            )
            regret = predicted_cost - oracle_cost

            realized_return = torch.sum(
                predicted_weights * true_returns,
                dim=-1,
            )
            equal_return = true_returns.mean(dim=-1)

            covariance_variance = torch.einsum(
                "bi,bij,bj->b",
                predicted_weights,
                covariance,
                predicted_weights,
            )
            effective_variance = torch.einsum(
                "bi,bij,bj->b",
                predicted_weights,
                effective_covariance,
                predicted_weights,
            )

            for sample_index, date in enumerate(dates):
                weight = predicted_weights[sample_index]

                turnover = (
                    0.0
                    if previous_weight is None
                    else 0.5
                    * torch.abs(weight - previous_weight).sum().item()
                )
                previous_weight = weight.clone()

                active_mask = weight > active_threshold
                capped_mask = (
                    weight >= max_weight - cap_tolerance
                )

                concentration = weight.square().sum().item()
                effective_asset_count = (
                    1.0 / concentration
                    if concentration > 0
                    else float("nan")
                )

                portfolio_return = realized_return[
                    sample_index
                ].item()
                equal_weight_return = equal_return[
                    sample_index
                ].item()

                portfolio_returns.append(portfolio_return)
                equal_weight_returns.append(equal_weight_return)

                daily_rows.append(
                    {
                        "date": date,
                        "portfolio_return": portfolio_return,
                        "equal_weight_return":
                            equal_weight_return,
                        "portfolio_variance":
                            covariance_variance[
                                sample_index
                            ].item(),
                        "effective_portfolio_variance":
                            effective_variance[
                                sample_index
                            ].item(),
                        "predicted_cost":
                            predicted_cost[
                                sample_index
                            ].item(),
                        "oracle_cost":
                            oracle_cost[
                                sample_index
                            ].item(),
                        "regret":
                            regret[
                                sample_index
                            ].item(),
                        "turnover": turnover,
                        "active_asset_count":
                            int(active_mask.sum().item()),
                        "capped_asset_count":
                            int(capped_mask.sum().item()),
                        "maximum_weight": weight.max().item(),
                        "minimum_weight": weight.min().item(),
                        "weight_sum": weight.sum().item(),
                        "weight_concentration": concentration,
                        "effective_asset_count":
                            effective_asset_count,
                    }
                )

                for asset_index, ticker in enumerate(tickers):
                    predicted_weight = weight[asset_index].item()
                    true_return = true_returns[
                        sample_index,
                        asset_index,
                    ].item()
                    predicted_return = predicted_returns[
                        sample_index,
                        asset_index,
                    ].item()
                    oracle_weight = oracle_weights[
                        sample_index,
                        asset_index,
                    ].item()

                    asset_rows.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "true_return": true_return,
                            "predicted_return":
                                predicted_return,
                            "prediction_bias":
                                predicted_return - true_return,
                            "predicted_weight":
                                predicted_weight,
                            "oracle_weight": oracle_weight,
                            "is_active":
                                predicted_weight
                                > active_threshold,
                            "is_capped":
                                predicted_weight
                                >= max_weight
                                - cap_tolerance,
                        }
                    )

    if not portfolio_returns:
        raise RuntimeError(
            "Test DataLoader에서 포트폴리오 수익률이 생성되지 않았습니다."
        )

    portfolio_tensor = torch.tensor(
        portfolio_returns,
        dtype=torch.float64,
    )
    equal_weight_tensor = torch.tensor(
        equal_weight_returns,
        dtype=torch.float64,
    )

    _, wealth, drawdown = calculate_portfolio_metrics(
        portfolio_tensor
    )
    _, equal_wealth, equal_drawdown = (
        calculate_portfolio_metrics(equal_weight_tensor)
    )

    daily = pd.DataFrame(daily_rows)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["wealth"] = wealth.numpy()
    daily["drawdown"] = drawdown.numpy()
    daily["equal_weight_wealth"] = equal_wealth.numpy()
    daily["equal_weight_drawdown"] = equal_drawdown.numpy()

    assets = pd.DataFrame(asset_rows)
    assets["date"] = pd.to_datetime(assets["date"])
    assets["active_weight"] = assets[
        "predicted_weight"
    ].where(assets["is_active"])

    asset_summary = (
        assets.groupby("ticker", as_index=False)
        .agg(
            average_weight=("predicted_weight", "mean"),
            median_weight=("predicted_weight", "median"),
            maximum_weight=("predicted_weight", "max"),
            minimum_weight=("predicted_weight", "min"),
            active_frequency=("is_active", "mean"),
            capped_frequency=("is_capped", "mean"),
            average_weight_when_active=(
                "active_weight",
                "mean",
            ),
            average_predicted_return=(
                "predicted_return",
                "mean",
            ),
            average_true_return=("true_return", "mean"),
            average_prediction_bias=(
                "prediction_bias",
                "mean",
            ),
        )
        .sort_values("average_weight", ascending=False)
        .reset_index(drop=True)
    )

    assets = assets.drop(columns=["active_weight"])

    return daily, assets, asset_summary


def load_baseline_daily(
    baseline_run_dir: str | Path,
) -> pd.DataFrame:
    path = Path(baseline_run_dir)

    if path.is_dir():
        path = path / "daily_portfolio.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"기존 DFL daily_portfolio.csv를 찾을 수 없습니다: {path}"
        )

    dataframe = pd.read_csv(path)

    required = {"date", "portfolio_return"}
    missing = required.difference(dataframe.columns)

    if missing:
        raise KeyError(
            f"기존 DFL 결과에 필요한 열이 없습니다: {sorted(missing)}"
        )

    selected_columns = ["date", "portfolio_return"]

    if "active_asset_count" in dataframe.columns:
        selected_columns.append("active_asset_count")

    dataframe = dataframe[selected_columns].copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe = dataframe.sort_values("date").reset_index(
        drop=True
    )
    dataframe = dataframe.rename(
        columns={
            "portfolio_return": "baseline_return",
            "active_asset_count":
                "baseline_active_asset_count",
        }
    )

    return dataframe


def build_comparison(
    daily: pd.DataFrame,
    n_assets: int,
    baseline_daily: pd.DataFrame | None,
    baseline_label: str,
    periods_per_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison = daily[
        [
            "date",
            "portfolio_return",
            "equal_weight_return",
            "active_asset_count",
            "capped_asset_count",
            "turnover",
            "effective_asset_count",
        ]
    ].rename(
        columns={
            "portfolio_return": "rcr_dfl_return",
        }
    )

    if baseline_daily is not None:
        comparison = comparison.merge(
            baseline_daily,
            on="date",
            how="inner",
            validate="one_to_one",
        )

    if comparison.empty:
        raise RuntimeError(
            "RCR-DFL과 비교 전략 사이에 공통 평가 날짜가 없습니다."
        )

    strategy_columns = [
        ("RCR-DFL", "rcr_dfl_return", "rcr_dfl"),
        (
            "Equal Weight",
            "equal_weight_return",
            "equal_weight",
        ),
    ]

    if baseline_daily is not None:
        strategy_columns.append(
            (
                baseline_label,
                "baseline_return",
                "baseline",
            )
        )

    summary_rows: list[dict[str, Any]] = []

    for strategy, return_column, prefix in strategy_columns:
        returns = torch.tensor(
            comparison[return_column].to_numpy(),
            dtype=torch.float64,
        )

        metrics, wealth, drawdown = (
            calculate_portfolio_metrics(
                returns,
                periods_per_year=periods_per_year,
            )
        )

        comparison[f"{prefix}_wealth"] = wealth.numpy()
        comparison[f"{prefix}_drawdown"] = (
            drawdown.numpy()
        )

        row: dict[str, Any] = {
            "strategy": strategy,
            "start_date":
                comparison["date"].iloc[0].date().isoformat(),
            "end_date":
                comparison["date"].iloc[-1].date().isoformat(),
            "n_observations": len(comparison),
            **metrics,
            "average_active_assets": float("nan"),
            "average_capped_assets": float("nan"),
            "average_daily_turnover": float("nan"),
            "average_effective_assets": float("nan"),
        }

        if prefix == "rcr_dfl":
            row.update(
                {
                    "average_active_assets": float(
                        comparison[
                            "active_asset_count"
                        ].mean()
                    ),
                    "average_capped_assets": float(
                        comparison[
                            "capped_asset_count"
                        ].mean()
                    ),
                    "average_daily_turnover": float(
                        comparison["turnover"].mean()
                    ),
                    "average_effective_assets": float(
                        comparison[
                            "effective_asset_count"
                        ].mean()
                    ),
                }
            )
        elif prefix == "equal_weight":
            row.update(
                {
                    "average_active_assets":
                        float(n_assets),
                    "average_capped_assets": 0.0,
                    "average_daily_turnover": 0.0,
                    "average_effective_assets":
                        float(n_assets),
                }
            )
        elif (
            prefix == "baseline"
            and "baseline_active_asset_count"
            in comparison.columns
        ):
            row["average_active_assets"] = float(
                comparison[
                    "baseline_active_asset_count"
                ].mean()
            )

        summary_rows.append(row)

    return comparison, pd.DataFrame(summary_rows)


def _save_line_plot(
    dataframe: pd.DataFrame,
    x_column: str,
    series: list[tuple[str, str]],
    title: str,
    y_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))

    for column, label in series:
        if column in dataframe.columns:
            axis.plot(
                dataframe[x_column],
                dataframe[column],
                label=label,
                linewidth=2.0,
            )

    axis.set_title(title)
    axis.set_xlabel(
        "Epoch" if x_column == "epoch" else "Date"
    )
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_loss_plots(
    history_path: Path,
    report_dir: Path,
    dpi: int,
) -> None:
    if not history_path.exists():
        raise FileNotFoundError(
            f"history.csv를 찾을 수 없습니다: {history_path}"
        )

    history = pd.read_csv(history_path)

    plot_specs = [
        (
            "total_loss",
            "Combined Loss",
            "loss_total.png",
        ),
        (
            "mse",
            "MSE",
            "loss_mse.png",
        ),
        (
            "regret",
            "RCR Regret",
            "loss_rcr.png",
        ),
        (
            "gradient_norm",
            "Gradient Norm",
            "gradient_norm.png",
        ),
    ]

    for metric, title, filename in plot_specs:
        train_column = f"train_{metric}"
        validation_column = f"validation_{metric}"

        if train_column not in history.columns:
            continue

        series = [(train_column, "Train")]

        if validation_column in history.columns:
            series.append(
                (validation_column, "Validation")
            )

        _save_line_plot(
            dataframe=history,
            x_column="epoch",
            series=series,
            title=title,
            y_label=title,
            output_path=report_dir / filename,
            dpi=dpi,
        )


def save_weight_plots(
    assets: pd.DataFrame,
    asset_summary: pd.DataFrame,
    report_dir: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    top_assets = asset_summary.head(15).sort_values(
        "average_weight",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        top_assets["ticker"],
        top_assets["average_weight"],
    )
    axis.set_title("Top 15 Assets by Average Weight")
    axis.set_xlabel("Average Weight")
    axis.set_ylabel("Ticker")
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        report_dir / "average_asset_weights.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    pivot = assets.pivot(
        index="date",
        columns="ticker",
        values="predicted_weight",
    ).sort_index()

    figure, axis = plt.subplots(figsize=(14, 8))
    image = axis.imshow(
        pivot.to_numpy().T,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_title("Test Portfolio Weight Heatmap")
    axis.set_ylabel("Ticker")
    axis.set_yticks(np.arange(len(pivot.columns)))
    axis.set_yticklabels(pivot.columns)

    date_count = len(pivot.index)
    tick_count = min(8, date_count)
    x_positions = np.linspace(
        0,
        date_count - 1,
        tick_count,
        dtype=int,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            pivot.index[position].strftime("%Y-%m-%d")
            for position in x_positions
        ],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Date")

    figure.colorbar(
        image,
        ax=axis,
        label="Portfolio Weight",
    )
    figure.tight_layout()
    figure.savefig(
        report_dir / "portfolio_weight_heatmap.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def generate_run_report(
    model: RCRMLPWithMarkowitz,
    test_loader: DataLoader,
    tickers: list[str],
    output_dir: str | Path,
    device: torch.device,
    metadata: dict[str, Any],
    best_epoch: int,
    best_validation_loss: float,
    baseline_run_dir: str | Path | None = None,
    baseline_label: str = "DFL-MVO",
    active_threshold: float = 1e-3,
    cap_tolerance: float = 1e-4,
    periods_per_year: int = 252,
    dpi: int = 300,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    report_dir = output_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    daily, assets, asset_summary = (
        evaluate_test_portfolio(
            model=model,
            test_loader=test_loader,
            tickers=tickers,
            device=device,
            active_threshold=active_threshold,
            cap_tolerance=cap_tolerance,
        )
    )

    baseline_daily = (
        load_baseline_daily(baseline_run_dir)
        if baseline_run_dir is not None
        else None
    )

    comparison, portfolio_summary = build_comparison(
        daily=daily,
        n_assets=len(tickers),
        baseline_daily=baseline_daily,
        baseline_label=baseline_label,
        periods_per_year=periods_per_year,
    )

    daily.to_csv(
        output_path / "daily_portfolio.csv",
        index=False,
        encoding="utf-8-sig",
    )
    assets.to_csv(
        output_path / "asset_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    asset_summary.to_csv(
        output_path / "asset_weight_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    portfolio_summary.to_csv(
        output_path / "portfolio_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        output_path / "comparison_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_loss_plots(
        history_path=output_path / "history.csv",
        report_dir=report_dir,
        dpi=dpi,
    )

    wealth_series = [
        ("rcr_dfl_wealth", "RCR-DFL"),
        ("equal_weight_wealth", "Equal Weight"),
    ]
    drawdown_series = [
        ("rcr_dfl_drawdown", "RCR-DFL"),
        (
            "equal_weight_drawdown",
            "Equal Weight",
        ),
    ]

    if baseline_daily is not None:
        wealth_series.append(
            ("baseline_wealth", baseline_label)
        )
        drawdown_series.append(
            ("baseline_drawdown", baseline_label)
        )

    _save_line_plot(
        dataframe=comparison,
        x_column="date",
        series=wealth_series,
        title="Out-of-Sample Cumulative Wealth",
        y_label="Cumulative Wealth",
        output_path=report_dir / "cumulative_wealth.png",
        dpi=dpi,
    )
    _save_line_plot(
        dataframe=comparison,
        x_column="date",
        series=drawdown_series,
        title="Out-of-Sample Drawdown",
        y_label="Drawdown",
        output_path=report_dir / "drawdown.png",
        dpi=dpi,
    )

    save_weight_plots(
        assets=assets,
        asset_summary=asset_summary,
        report_dir=report_dir,
        dpi=dpi,
    )

    latest_date = assets["date"].max()
    latest_weights = (
        assets.loc[assets["date"] == latest_date]
        .sort_values(
            "predicted_weight",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    latest_weights.to_csv(
        output_path / "latest_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    strategies_payload = {
        str(row["strategy"]): {
            key: value
            for key, value in row.items()
            if key != "strategy"
        }
        for row in portfolio_summary.to_dict(
            orient="records"
        )
    }

    summary_payload = {
        "experiment": {
            "alpha": metadata.get("alpha"),
            "eta": metadata.get("eta"),
            "rho": metadata.get(
                "residual_correlation_shrinkage"
            ),
            "lambda": metadata.get("risk_aversion"),
            "max_weight": metadata.get("max_weight"),
            "mse_scale": metadata.get("mse_scale"),
            "seed": metadata.get("seed"),
            "standardize_inputs": metadata.get(
                "standardize_inputs",
                False,
            ),
            "best_epoch": int(best_epoch),
            "best_validation_loss":
                float(best_validation_loss),
        },
        "evaluation": {
            "start_date":
                comparison["date"].iloc[0]
                .date()
                .isoformat(),
            "end_date":
                comparison["date"].iloc[-1]
                .date()
                .isoformat(),
            "n_observations": len(comparison),
            "periods_per_year": periods_per_year,
            "baseline_run_dir": (
                str(Path(baseline_run_dir))
                if baseline_run_dir is not None
                else None
            ),
            "baseline_label": (
                baseline_label
                if baseline_run_dir is not None
                else None
            ),
        },
        "strategies": strategies_payload,
    }

    save_json(
        output_path / "summary.json",
        summary_payload,
    )

    manifest = {
        "csv_files": [
            "daily_portfolio.csv",
            "asset_predictions.csv",
            "asset_weight_summary.csv",
            "latest_weights.csv",
            "portfolio_summary.csv",
            "comparison_daily.csv",
        ],
        "plot_files": [
            "report/loss_total.png",
            "report/loss_mse.png",
            "report/loss_rcr.png",
            "report/gradient_norm.png",
            "report/cumulative_wealth.png",
            "report/drawdown.png",
            "report/average_asset_weights.png",
            "report/portfolio_weight_heatmap.png",
        ],
    }
    save_json(
        output_path / "report_manifest.json",
        manifest,
    )

    display_columns = [
        "strategy",
        "annualized_return_cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_wealth",
        "average_active_assets",
    ]

    print()
    print("=" * 100)
    print("Portfolio Report")
    print("=" * 100)
    print(
        portfolio_summary[display_columns]
        .to_string(index=False)
    )
    print()
    print("Top 10 assets by average weight")
    print(
        asset_summary[
            [
                "ticker",
                "average_weight",
                "maximum_weight",
                "active_frequency",
                "capped_frequency",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print()
    print(f"Report saved: {output_path}")

    return portfolio_summary
