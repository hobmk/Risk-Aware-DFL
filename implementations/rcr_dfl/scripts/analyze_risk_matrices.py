from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset
from implementations.rcr_dfl.src.effective_covariance import build_effective_covariance
from implementations.rcr_dfl.src.residual_risk import matrix_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sigma, residual correlation, A_res, Sigma_eff 비교")
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--market-mode", choices=["equal_weight", "external"], default="equal_weight")
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--return-type", choices=["simple", "log"], default="simple")
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--residual-correlation-shrinkage", type=float, default=0.1)
    parser.add_argument("--correlation-scaling", choices=["none", "trace"], default="trace")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--effective-jitter", type=float, default=0.0)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--output-dir", default="implementations/rcr_dfl/outputs/risk_matrix_analysis")
    return parser.parse_args()


def resolve_index(index: int, length: int) -> int:
    resolved = index if index >= 0 else length + index
    if not 0 <= resolved < length:
        raise IndexError(f"sample-index가 범위를 벗어났습니다: {index}, samples={length}")
    return resolved


def save_heatmap(matrix: torch.Tensor, tickers: list[str], title: str, path: Path) -> None:
    values = matrix.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(values, aspect="auto")
    axis.set_title(title)
    step = max(1, len(tickers) // 15)
    positions = np.arange(0, len(tickers), step)
    axis.set_xticks(positions, [tickers[i] for i in positions], rotation=90)
    axis.set_yticks(positions, [tickers[i] for i in positions])
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def off_diagonal_values(matrices: torch.Tensor) -> np.ndarray:
    n_assets = matrices.size(-1)
    mask = ~torch.eye(n_assets, dtype=torch.bool, device=matrices.device)
    return matrices[..., mask].detach().cpu().numpy().reshape(-1)


def save_distribution(
    first: np.ndarray,
    second: np.ndarray,
    first_label: str,
    second_label: str,
    title: str,
    bins: int,
    path: Path,
) -> None:
    combined = np.concatenate([first, second])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        raise ValueError("분포 그래프를 생성할 유한한 값이 없습니다.")
    limits = np.quantile(finite, [0.005, 0.995])
    if limits[0] == limits[1]:
        limits = np.array([finite.min(), finite.max()])
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.hist(first, bins=bins, range=tuple(limits), density=True, alpha=0.55, label=first_label)
    axis.hist(second, bins=bins, range=tuple(limits), density=True, alpha=0.55, label=second_label)
    axis.set_title(title)
    axis.set_xlabel("Off-diagonal value")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def diagnostics_rows(
    name: str,
    matrices: torch.Tensor,
    dates: pd.DatetimeIndex,
) -> list[dict[str, float | str | bool]]:
    diagnostics = matrix_diagnostics(matrices)
    rows = []
    for index, date in enumerate(dates):
        condition = diagnostics.condition_number[index].item()
        minimum = diagnostics.minimum_eigenvalue[index].item()
        rows.append(
            {
                "date": str(date.date()),
                "matrix": name,
                "symmetry_error": diagnostics.symmetry_error[index].item(),
                "minimum_eigenvalue": minimum,
                "maximum_eigenvalue": diagnostics.maximum_eigenvalue[index].item(),
                "condition_number": condition,
                "is_psd_1e-10": minimum >= -1e-10,
                "condition_is_finite": bool(np.isfinite(condition)),
            }
        )
    return rows


def summarize_diagnostics(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for matrix_name, group in dataframe.groupby("matrix", sort=False):
        finite_condition = group.loc[np.isfinite(group["condition_number"]), "condition_number"]
        rows.append(
            {
                "matrix": matrix_name,
                "max_symmetry_error": group["symmetry_error"].max(),
                "minimum_eigenvalue": group["minimum_eigenvalue"].min(),
                "median_minimum_eigenvalue": group["minimum_eigenvalue"].median(),
                "psd_fraction": group["is_psd_1e-10"].mean(),
                "median_condition_number": finite_condition.median() if not finite_condition.empty else np.inf,
                "maximum_condition_number": finite_condition.max() if not finite_condition.empty else np.inf,
                "finite_condition_fraction": group["condition_is_finite"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RCRRollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type=args.return_type,
        covariance_jitter=args.covariance_jitter,
        market_mode=args.market_mode,
        market_price_csv=args.market_price_csv,
        market_column=args.market_column,
        risk_free_rate=args.risk_free_rate,
        residual_correlation_shrinkage=args.residual_correlation_shrinkage,
        correlation_scaling=args.correlation_scaling,
    )
    effective_covariances = build_effective_covariance(
        covariance=dataset.covariances,
        residual_risk_matrix=dataset.a_res_matrices,
        eta=args.eta,
        jitter=args.effective_jitter,
    )
    sample_index = resolve_index(args.sample_index, len(dataset))
    sample_date = dataset.target_dates[sample_index]

    matrices = {
        "Sigma": dataset.covariances,
        "Residual correlation raw": dataset.residual_correlations_raw,
        "Residual correlation shrunk": dataset.residual_correlations,
        "A_res": dataset.a_res_matrices,
        "Sigma_eff": effective_covariances,
    }
    safe_names = {
        "Sigma": "covariance",
        "Residual correlation raw": "residual_correlation_raw",
        "Residual correlation shrunk": "residual_correlation_shrunk",
        "A_res": "a_res",
        "Sigma_eff": "effective_covariance",
    }
    for name, matrix in matrices.items():
        save_heatmap(
            matrix[sample_index],
            dataset.tickers,
            f"{name} | {sample_date.date()}",
            output_dir / f"heatmap_{safe_names[name]}.png",
        )

    save_distribution(
        off_diagonal_values(dataset.covariances),
        off_diagonal_values(dataset.a_res_matrices),
        "Sigma",
        "A_res",
        "All rolling windows: Sigma vs A_res off-diagonal distribution",
        args.bins,
        output_dir / "distribution_sigma_vs_a_res.png",
    )
    save_distribution(
        off_diagonal_values(dataset.residual_correlations_raw),
        off_diagonal_values(dataset.residual_correlations),
        "Raw residual correlation",
        "Shrunk residual correlation",
        "All rolling windows: residual correlation distribution",
        args.bins,
        output_dir / "distribution_residual_correlation.png",
    )

    rows: list[dict[str, float | str | bool]] = []
    for name, matrix in matrices.items():
        rows.extend(diagnostics_rows(name, matrix, dataset.target_dates))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output_dir / "matrix_diagnostics_by_date.csv", index=False)
    summary = summarize_diagnostics(diagnostics)
    summary.to_csv(output_dir / "matrix_diagnostics_summary.csv", index=False)

    np.savez_compressed(
        output_dir / "selected_sample_matrices.npz",
        covariance=dataset.covariances[sample_index].cpu().numpy(),
        residual_correlation_raw=dataset.residual_correlations_raw[sample_index].cpu().numpy(),
        residual_correlation=dataset.residual_correlations[sample_index].cpu().numpy(),
        a_res=dataset.a_res_matrices[sample_index].cpu().numpy(),
        effective_covariance=effective_covariances[sample_index].cpu().numpy(),
        tickers=np.asarray(dataset.tickers),
        date=str(sample_date.date()),
    )
    config = vars(args).copy()
    config.update(
        {
            "resolved_sample_index": sample_index,
            "resolved_sample_date": str(sample_date.date()),
            "n_assets": dataset.n_assets,
            "n_samples": len(dataset),
            "market_name": dataset.market_name,
        }
    )
    with (output_dir / "analysis_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(summary.to_string(index=False))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
