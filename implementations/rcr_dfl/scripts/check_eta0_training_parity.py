from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
    chronological_split as dfl_split,
    fit_feature_standardizer as dfl_fit_standardizer,
    FeatureStandardizedSubset as DFLStandardizedSubset,
)
from implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_combined import (
    MLPWithMarkowitz,
    compute_losses,
)

from implementations.rcr_dfl.src.dataset import (
    RCRRollingMVODataset,
    chronological_split as rcr_split,
    fit_feature_standardizer as rcr_fit_standardizer,
    FeatureStandardizedSubset as RCRStandardizedSubset,
)
from implementations.rcr_dfl.src.decision_model import RCRMLPWithMarkowitz
from implementations.rcr_dfl.src.losses import compute_rcr_losses


PRICE_CSV = PROJECT_ROOT / "data/raw/dow30_adjusted_close.csv"

LOOKBACK = 60
HIDDEN_DIM = 256
DROPOUT = 0.0

RISK_AVERSION = 1.0
MAX_WEIGHT = 0.2

ALPHA = 0.5
MSE_SCALE = 30.0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

SEED = 42
BATCH_SIZE = 16
N_STEPS = 10


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(device="cpu", dtype=torch.float64)
    b = b.detach().to(device="cpu", dtype=torch.float64)
    return (a - b).abs().max().item()


def stack(dataset, indices, key: str) -> torch.Tensor:
    return torch.stack(
        [dataset[int(i)][key] for i in indices],
        dim=0,
    )


def max_parameter_diff(model_a, model_b) -> float:
    maximum = 0.0

    parameters_a = dict(model_a.return_model.named_parameters())
    parameters_b = dict(model_b.return_model.named_parameters())

    for name in parameters_a:
        difference = max_diff(
            parameters_a[name],
            parameters_b[name],
        )
        maximum = max(maximum, difference)

    return maximum


def max_gradient_diff(model_a, model_b) -> float:
    maximum = 0.0

    parameters_a = dict(model_a.return_model.named_parameters())
    parameters_b = dict(model_b.return_model.named_parameters())

    for name in parameters_a:
        gradient_a = parameters_a[name].grad
        gradient_b = parameters_b[name].grad

        if gradient_a is None and gradient_b is None:
            continue

        if gradient_a is None or gradient_b is None:
            return float("inf")

        difference = max_diff(
            gradient_a,
            gradient_b,
        )
        maximum = max(maximum, difference)

    return maximum


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("DFL-MVO vs RCR-DFL eta=0 TRAINING PARITY TEST")
    print("=" * 100)
    print("device:", device)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dfl_dataset = RollingMVODataset(
        price_csv=PRICE_CSV,
        lookback=LOOKBACK,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float32,
    )

    rcr_dataset = RCRRollingMVODataset(
        price_csv=PRICE_CSV,
        lookback=LOOKBACK,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        residual_correlation_shrinkage=0.0,
        correlation_scaling="trace",
        dtype=torch.float32,
    )

    dfl_train, _, _ = dfl_split(
        dfl_dataset,
        train_end="2021-12-31",
        validation_end="2022-12-31",
    )

    rcr_train, _, _ = rcr_split(
        rcr_dataset,
        train_end="2021-12-31",
        validation_end="2022-12-31",
    )

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------
    dfl_mean, dfl_std = dfl_fit_standardizer(
        dfl_dataset,
        dfl_train,
    )

    rcr_mean, rcr_std = rcr_fit_standardizer(
        rcr_dataset,
        rcr_train,
    )

    dfl_train = DFLStandardizedSubset(
        dfl_train,
        dfl_mean,
        dfl_std,
    )

    rcr_train = RCRStandardizedSubset(
        rcr_train,
        rcr_mean,
        rcr_std,
    )

    # ------------------------------------------------------------------
    # 동일한 random sample 순서 생성
    # ------------------------------------------------------------------
    generator = torch.Generator().manual_seed(SEED)

    permutation = torch.randperm(
        len(dfl_train),
        generator=generator,
    )

    required_samples = BATCH_SIZE * N_STEPS

    if required_samples > len(permutation):
        raise RuntimeError("Training sample 수가 부족합니다.")

    permutation = permutation[:required_samples]

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)

    dfl_model = MLPWithMarkowitz(
        n_assets=dfl_dataset.n_assets,
        lookback=LOOKBACK,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        risk_aversion=RISK_AVERSION,
        max_weight=MAX_WEIGHT,
    ).float()

    rcr_model = RCRMLPWithMarkowitz(
        n_assets=rcr_dataset.n_assets,
        lookback=LOOKBACK,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        risk_aversion=RISK_AVERSION,
        max_weight=MAX_WEIGHT,
        eta=0.0,
        effective_jitter=0.0,
        project_psd=False,
        minimum_eigenvalue=0.0,
    ).float()

    # 동일한 초기 MLP parameter 강제
    rcr_model.return_model.load_state_dict(
        dfl_model.return_model.state_dict()
    )

    # optimizer 생성 전에 device 이동
    dfl_model = dfl_model.to(device)
    rcr_model = rcr_model.to(device)

    optimizer_dfl = torch.optim.AdamW(
        dfl_model.return_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_rcr = torch.optim.AdamW(
        rcr_model.return_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "\nInitial parameter diff:",
        f"{max_parameter_diff(dfl_model, rcr_model):.12e}",
    )

    print("\n" + "-" * 100)
    print(
        f"{'Step':>4} | "
        f"{'Loss DFL':>14} | "
        f"{'Loss RCR':>14} | "
        f"{'Loss diff':>12} | "
        f"{'Grad diff':>12} | "
        f"{'Param diff':>12}"
    )
    print("-" * 100)

    # ------------------------------------------------------------------
    # 10 optimizer steps
    # ------------------------------------------------------------------
    print(
        f"\nDEBUG BEFORE LOOP | "
        f"N_STEPS={N_STEPS} | "
        f"BATCH_SIZE={BATCH_SIZE} | "
        f"required_samples={required_samples} | "
        f"permutation_len={len(permutation)} | "
        f"range={list(range(N_STEPS))}",
        flush=True,
    )

    for step in range(N_STEPS):
        print(f"DEBUG ENTER STEP {step + 1}", flush=True)

        start = step * BATCH_SIZE
        end = start + BATCH_SIZE
        indices = permutation[start:end]

        features_dfl = stack(
            dfl_train,
            indices,
            "features",
        ).to(device)

        target_dfl = stack(
            dfl_train,
            indices,
            "target",
        ).to(device)

        covariance_dfl = stack(
            dfl_train,
            indices,
            "covariance",
        )

        features_rcr = stack(
            rcr_train,
            indices,
            "features",
        ).to(device)

        target_rcr = stack(
            rcr_train,
            indices,
            "target",
        ).to(device)

        covariance_rcr = stack(
            rcr_train,
            indices,
            "covariance",
        )

        a_res = stack(
            rcr_train,
            indices,
            "a_res",
        )

        # --------------------------------------------------------------
        # DFL
        # --------------------------------------------------------------
        optimizer_dfl.zero_grad(set_to_none=True)

        (
            total_dfl,
            regret_dfl,
            mse_dfl,
            predicted_weights_dfl,
            risk_factor_dfl,
        ) = compute_losses(
            model=dfl_model,
            features=features_dfl,
            targets=target_dfl,
            covariance=covariance_dfl,
            alpha=ALPHA,
            mse_scale=MSE_SCALE,
        )

        total_dfl.backward()

        # --------------------------------------------------------------
        # RCR eta=0
        # --------------------------------------------------------------
        optimizer_rcr.zero_grad(set_to_none=True)

        output_rcr = rcr_model(
            features=features_rcr,
            covariance=covariance_rcr,
            a_res=a_res,
        )

        oracle_weights_rcr = rcr_model.solve_oracle(
            true_returns=target_rcr,
            risk_factor=output_rcr.risk_factor,
        )

        losses_rcr = compute_rcr_losses(
            predicted_returns=output_rcr.predicted_returns,
            predicted_weights=output_rcr.predicted_weights,
            oracle_weights=oracle_weights_rcr,
            true_returns=target_rcr,
            risk_factor=output_rcr.risk_factor,
            alpha=ALPHA,
            mse_scale=MSE_SCALE,
        )

        losses_rcr.total.backward()

        # optimizer 적용 전 gradient 비교
        gradient_difference = max_gradient_diff(
            dfl_model,
            rcr_model,
        )

        loss_difference = abs(
            total_dfl.item()
            - losses_rcr.total.item()
        )

        # --------------------------------------------------------------
        # 동일 optimizer step
        # --------------------------------------------------------------
        optimizer_dfl.step()
        optimizer_rcr.step()

        parameter_difference = max_parameter_diff(
            dfl_model,
            rcr_model,
        )

        print(
            f"{step + 1:>4} | "
            f"{total_dfl.item():>14.10f} | "
            f"{losses_rcr.total.item():>14.10f} | "
            f"{loss_difference:>12.3e} | "
            f"{gradient_difference:>12.3e} | "
            f"{parameter_difference:>12.3e}",
            flush=True,
        )

    print("-" * 100)

    final_difference = max_parameter_diff(
        dfl_model,
        rcr_model,
    )

    print(
        "\nFinal maximum parameter difference:",
        f"{final_difference:.12e}",
    )

    print("\n" + "=" * 100)

    if final_difference < 1e-7:
        print("PASS: eta=0 training parity 확인")
    else:
        print("CHECK: optimizer step 이후 두 모델이 달라집니다.")

    print("=" * 100)


if __name__ == "__main__":
    main()