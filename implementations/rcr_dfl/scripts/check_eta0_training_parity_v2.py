from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
    chronological_split as dfl_split,
    fit_feature_standardizer as dfl_fit_standardizer,
    FeatureStandardizedSubset as DFLStandardizedSubset,
)
from implementations.dfl_mvo_lee2025.src.losses import (
    mvo_regret,
    combined_loss as dfl_combined_loss,
)
from implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_combined import (
    MLPWithMarkowitz,
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
HIDDEN_DIM = 64
DROPOUT = 0.0

RISK_AVERSION = 1.0
MAX_WEIGHT = 0.1

ALPHA = 0.5
MSE_SCALE = 30.0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

SEED = 42
BATCH_SIZE = 16
N_STEPS = 5


def max_diff(a, b):
    a = a.detach().cpu().double()
    b = b.detach().cpu().double()
    return (a - b).abs().max().item()


def max_parameter_diff(model_a, model_b):
    max_value = 0.0
    params_a = dict(model_a.return_model.named_parameters())
    params_b = dict(model_b.return_model.named_parameters())

    for name in params_a:
        max_value = max(
            max_value,
            max_diff(params_a[name], params_b[name]),
        )

    return max_value


def max_gradient_diff(model_a, model_b):
    max_value = 0.0
    params_a = dict(model_a.return_model.named_parameters())
    params_b = dict(model_b.return_model.named_parameters())

    for name in params_a:
        grad_a = params_a[name].grad
        grad_b = params_b[name].grad

        if grad_a is None or grad_b is None:
            raise RuntimeError(f"Gradient missing: {name}")

        max_value = max(
            max_value,
            max_diff(grad_a, grad_b),
        )

    return max_value


def get_batch(dataset, indices, key):
    return torch.stack(
        [dataset[int(i)][key] for i in indices],
        dim=0,
    )


def main():
    print("=" * 90)
    print("ETA=0 TRAINING PARITY V2")
    print("=" * 90)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device, flush=True)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 동일 sample 순서
    # ------------------------------------------------------------
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(
        len(dfl_train),
        generator=generator,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
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

    # 동일 초기 weight
    rcr_model.return_model.load_state_dict(
        dfl_model.return_model.state_dict()
    )

    dfl_model.to(device)
    rcr_model.to(device)

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
        "initial parameter diff:",
        f"{max_parameter_diff(dfl_model, rcr_model):.12e}",
        flush=True,
    )

    print(
        f"\nSTART LOOP | N_STEPS={N_STEPS}",
        flush=True,
    )

    # ============================================================
    # Training parity
    # ============================================================
    for step in range(N_STEPS):
        print(
            f"\n===== STEP {step + 1}/{N_STEPS} =====",
            flush=True,
        )

        start = step * BATCH_SIZE
        end = start + BATCH_SIZE
        indices = permutation[start:end]

        features_dfl = get_batch(
            dfl_train, indices, "features"
        ).to(device)

        target_dfl = get_batch(
            dfl_train, indices, "target"
        ).to(device)

        covariance_dfl = get_batch(
            dfl_train, indices, "covariance"
        )

        features_rcr = get_batch(
            rcr_train, indices, "features"
        ).to(device)

        target_rcr = get_batch(
            rcr_train, indices, "target"
        ).to(device)

        covariance_rcr = get_batch(
            rcr_train, indices, "covariance"
        )

        a_res = get_batch(
            rcr_train, indices, "a_res"
        )

        # --------------------------------------------------------
        # 입력 parity 확인
        # --------------------------------------------------------
        print(
            "input diff:",
            f"{max_diff(features_dfl, features_rcr):.3e}",
            "cov diff:",
            f"{max_diff(covariance_dfl, covariance_rcr):.3e}",
            flush=True,
        )

        # --------------------------------------------------------
        # DFL forward/loss
        # --------------------------------------------------------
        optimizer_dfl.zero_grad(set_to_none=True)

        (
            pred_returns_dfl,
            pred_weights_dfl,
            risk_factor_dfl,
        ) = dfl_model(
            features=features_dfl,
            covariance=covariance_dfl,
        )

        target_solver_dfl = target_dfl.to(
            device="cpu",
            dtype=torch.float64,
        )

        oracle_dfl, = dfl_model.markowitz_layer(
            target_solver_dfl,
            risk_factor_dfl,
        )

        oracle_dfl = oracle_dfl.detach()

        regret_dfl = mvo_regret(
            predicted_weights=pred_weights_dfl,
            oracle_weights=oracle_dfl,
            true_returns=target_solver_dfl,
            risk_factor=risk_factor_dfl,
            reduction="mean",
        )

        mse_dfl = F.mse_loss(
            pred_returns_dfl,
            target_dfl,
        )

        total_dfl = dfl_combined_loss(
            regret_loss=regret_dfl,
            mse_loss=mse_dfl,
            alpha=ALPHA,
            mse_scale=MSE_SCALE,
        )

        total_dfl.backward()

        # --------------------------------------------------------
        # RCR forward/loss
        # --------------------------------------------------------
        optimizer_rcr.zero_grad(set_to_none=True)

        output_rcr = rcr_model(
            features=features_rcr,
            covariance=covariance_rcr,
            a_res=a_res,
        )

        oracle_rcr = rcr_model.solve_oracle(
            true_returns=target_rcr,
            risk_factor=output_rcr.risk_factor,
        )

        losses_rcr = compute_rcr_losses(
            predicted_returns=output_rcr.predicted_returns,
            predicted_weights=output_rcr.predicted_weights,
            oracle_weights=oracle_rcr,
            true_returns=target_rcr,
            risk_factor=output_rcr.risk_factor,
            alpha=ALPHA,
            mse_scale=MSE_SCALE,
        )

        losses_rcr.total.backward()

        # --------------------------------------------------------
        # optimizer 전 비교
        # --------------------------------------------------------
        loss_diff = abs(
            total_dfl.item()
            - losses_rcr.total.item()
        )

        grad_diff = max_gradient_diff(
            dfl_model,
            rcr_model,
        )

        print(
            f"DFL loss  : {total_dfl.item():.12e}"
        )
        print(
            f"RCR loss  : {losses_rcr.total.item():.12e}"
        )
        print(
            f"loss diff : {loss_diff:.12e}"
        )
        print(
            f"grad diff : {grad_diff:.12e}"
        )

        # --------------------------------------------------------
        # AdamW
        # --------------------------------------------------------
        optimizer_dfl.step()
        optimizer_rcr.step()

        param_diff = max_parameter_diff(
            dfl_model,
            rcr_model,
        )

        print(
            f"param diff: {param_diff:.12e}",
            flush=True,
        )

    final_diff = max_parameter_diff(
        dfl_model,
        rcr_model,
    )

    print("\n" + "=" * 90)
    print(
        "FINAL PARAMETER DIFF:",
        f"{final_diff:.12e}",
    )

    if final_diff < 1e-7:
        print("PASS: eta=0 training parity")
    else:
        print("FAIL: training paths diverged")

    print("=" * 90)


if __name__ == "__main__":
    main()