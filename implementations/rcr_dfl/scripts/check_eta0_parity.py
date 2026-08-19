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
from implementations.rcr_dfl.src.decision_model import (
    RCRMLPWithMarkowitz,
)
from implementations.rcr_dfl.src.losses import (
    compute_rcr_losses,
)


PRICE_CSV = PROJECT_ROOT / "data/raw/dow30_adjusted_close.csv"

LOOKBACK = 60
HIDDEN_DIM = 256
DROPOUT = 0.0

RISK_AVERSION = 1.0
MAX_WEIGHT = 0.2

ALPHA = 0.5
MSE_SCALE = 30.0

SEED = 42
BATCH_SIZE = 4


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(device="cpu", dtype=torch.float64)
    b = b.detach().to(device="cpu", dtype=torch.float64)
    return (a - b).abs().max().item()


def print_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
    diff = max_diff(a, b)
    print(f"{name:<30}: {diff:.12e}")
    return diff


def stack(dataset, key: str) -> torch.Tensor:
    return torch.stack(
        [dataset[i][key] for i in range(BATCH_SIZE)],
        dim=0,
    )


def main() -> None:
    torch.manual_seed(SEED)

    print("=" * 80)
    print("DFL-MVO vs RCR-DFL eta=0 PARITY TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    print("\n[1] Dataset 생성")

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

    print(f"DFL samples : {len(dfl_dataset)}")
    print(f"RCR samples : {len(rcr_dataset)}")
    print(f"Ticker equal: {dfl_dataset.tickers == rcr_dataset.tickers}")
    print(
        "Date equal  :",
        dfl_dataset.target_dates.equals(rcr_dataset.target_dates),
    )

    if len(dfl_dataset) != len(rcr_dataset):
        raise RuntimeError("Dataset sample 수가 다릅니다.")

    if dfl_dataset.tickers != rcr_dataset.tickers:
        raise RuntimeError("Ticker 순서가 다릅니다.")

    if not dfl_dataset.target_dates.equals(rcr_dataset.target_dates):
        raise RuntimeError("Target date가 다릅니다.")

    # ------------------------------------------------------------------
    # 2. Raw sample parity
    # ------------------------------------------------------------------
    print("\n[2] Raw Dataset parity")

    for index in [0, 100, 500, 1000]:
        if index >= len(dfl_dataset):
            continue

        dfl_sample = dfl_dataset[index]
        rcr_sample = rcr_dataset[index]

        print(f"\nSample {index} | {dfl_sample['target_date']}")

        print_diff(
            "features",
            dfl_sample["features"],
            rcr_sample["features"],
        )
        print_diff(
            "target",
            dfl_sample["target"],
            rcr_sample["target"],
        )
        print_diff(
            "covariance",
            dfl_sample["covariance"],
            rcr_sample["covariance"],
        )

    # ------------------------------------------------------------------
    # 3. 동일 Train Split
    # ------------------------------------------------------------------
    print("\n[3] Train split / Standardization")

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

    print(f"DFL train samples: {len(dfl_train)}")
    print(f"RCR train samples: {len(rcr_train)}")

    dfl_mean, dfl_std = dfl_fit_standardizer(
        dfl_dataset,
        dfl_train,
    )

    rcr_mean, rcr_std = rcr_fit_standardizer(
        rcr_dataset,
        rcr_train,
    )

    print_diff("feature mean", dfl_mean, rcr_mean)
    print_diff("feature std", dfl_std, rcr_std)

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
    # 4. 동일 batch
    # ------------------------------------------------------------------
    features_dfl = stack(dfl_train, "features")
    features_rcr = stack(rcr_train, "features")

    target_dfl = stack(dfl_train, "target")
    target_rcr = stack(rcr_train, "target")

    covariance_dfl = stack(dfl_train, "covariance")
    covariance_rcr = stack(rcr_train, "covariance")

    a_res = stack(rcr_train, "a_res")

    print("\n[4] Standardized batch parity")

    print_diff(
        "features",
        features_dfl,
        features_rcr,
    )
    print_diff(
        "target",
        target_dfl,
        target_rcr,
    )
    print_diff(
        "covariance",
        covariance_dfl,
        covariance_rcr,
    )

    # ------------------------------------------------------------------
    # 5. Model 생성
    # ------------------------------------------------------------------
    print("\n[5] Model parity")

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

    # 중요:
    # 두 모델이 완전히 동일한 MLP weight로 시작하도록 강제로 복사
    rcr_model.return_model.load_state_dict(
        dfl_model.return_model.state_dict()
    )

    # ------------------------------------------------------------------
    # 6. DFL forward
    # ------------------------------------------------------------------
    (
        predicted_returns_dfl,
        predicted_weights_dfl,
        risk_factor_dfl,
    ) = dfl_model(
        features=features_dfl,
        covariance=covariance_dfl,
    )

    target_dfl_solver = target_dfl.to(
        device="cpu",
        dtype=torch.float64,
    )

    oracle_weights_dfl, = dfl_model.markowitz_layer(
        target_dfl_solver,
        risk_factor_dfl,
    )

    oracle_weights_dfl = oracle_weights_dfl.detach()

    regret_dfl = mvo_regret(
        predicted_weights=predicted_weights_dfl,
        oracle_weights=oracle_weights_dfl,
        true_returns=target_dfl_solver,
        risk_factor=risk_factor_dfl,
        reduction="mean",
    )

    mse_dfl = F.mse_loss(
        predicted_returns_dfl,
        target_dfl,
    )

    total_dfl = dfl_combined_loss(
        regret_loss=regret_dfl,
        mse_loss=mse_dfl,
        alpha=ALPHA,
        mse_scale=MSE_SCALE,
    )

    # ------------------------------------------------------------------
    # 7. RCR forward, eta=0
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 8. Forward 결과 비교
    # ------------------------------------------------------------------
    print("\n[6] Forward / Optimization parity")

    print_diff(
        "predicted returns",
        predicted_returns_dfl,
        output_rcr.predicted_returns,
    )

    print_diff(
        "Sigma vs Sigma_eff",
        covariance_dfl.to(torch.float64),
        output_rcr.effective_covariance,
    )

    print_diff(
        "risk factor",
        risk_factor_dfl,
        output_rcr.risk_factor,
    )

    print_diff(
        "predicted weights",
        predicted_weights_dfl,
        output_rcr.predicted_weights,
    )

    print_diff(
        "oracle weights",
        oracle_weights_dfl,
        oracle_weights_rcr,
    )

    print("\nLoss")

    print(f"DFL MSE       : {mse_dfl.item():.12e}")
    print(f"RCR MSE       : {losses_rcr.mse.item():.12e}")
    print(
        f"MSE diff      : "
        f"{abs(mse_dfl.item() - losses_rcr.mse.item()):.12e}"
    )

    print(f"DFL regret    : {regret_dfl.item():.12e}")
    print(f"RCR regret    : {losses_rcr.regret.item():.12e}")
    print(
        f"Regret diff   : "
        f"{abs(regret_dfl.item() - losses_rcr.regret.item()):.12e}"
    )

    print(f"DFL combined  : {total_dfl.item():.12e}")
    print(f"RCR combined  : {losses_rcr.total.item():.12e}")
    print(
        f"Combined diff : "
        f"{abs(total_dfl.item() - losses_rcr.total.item()):.12e}"
    )

    # ------------------------------------------------------------------
    # 9. Gradient parity
    # ------------------------------------------------------------------
    print("\n[7] Gradient parity")

    dfl_model.zero_grad(set_to_none=True)
    rcr_model.zero_grad(set_to_none=True)

    total_dfl.backward()
    losses_rcr.total.backward()

    max_gradient_diff = 0.0

    dfl_parameters = dict(
        dfl_model.return_model.named_parameters()
    )
    rcr_parameters = dict(
        rcr_model.return_model.named_parameters()
    )

    for name in dfl_parameters:
        grad_dfl = dfl_parameters[name].grad
        grad_rcr = rcr_parameters[name].grad

        if grad_dfl is None or grad_rcr is None:
            print(f"{name:<30}: gradient=None")
            continue

        diff = max_diff(
            grad_dfl,
            grad_rcr,
        )

        max_gradient_diff = max(
            max_gradient_diff,
            diff,
        )

        print(
            f"{name:<30}: "
            f"{diff:.12e}"
        )

    print(
        "\nMAX gradient diff:",
        f"{max_gradient_diff:.12e}",
    )

    print("\n" + "=" * 80)
    print("PARITY TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()