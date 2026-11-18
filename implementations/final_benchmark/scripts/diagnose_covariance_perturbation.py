
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset, chronological_split, fit_feature_standardizer,
)
from implementations.covariance_baselines.src.dataset import CovarianceBaselineDataset
from implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_combined import MLPWithMarkowitz
from implementations.dfl_mvo_lee2025.src.losses import markowitz_cost


def parse_args():
    p = argparse.ArgumentParser(description="Fixed-predictor covariance perturbation diagnostic")
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--out-dir", default="implementations/final_benchmark/outputs/diagnostics/perturbation_fixed_model")
    p.add_argument("--checkpoint-root", default="implementations/final_benchmark/outputs")
    p.add_argument("--predictor-source", choices=["dfl_mvo", "ridge"], default="dfl_mvo")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    p.add_argument("--quantiles", nargs="+", type=float, default=[0.5, 0.9, 0.99])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mse-scale", type=float, default=30.0)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--direction-seed", type=int, default=20261110)
    p.add_argument("--active-tolerance", type=float, default=1e-4)
    p.add_argument("--skip-aa", action="store_true")
    return p.parse_args()


def norm(x):
    return float(np.linalg.norm(x))


def similarity(x, y):
    a, b = norm(x), norm(y)
    return float(np.dot(x, y) / (a * b)) if a > 0 and b > 0 else float("nan")


def flatten(grads):
    return np.concatenate([g.detach().cpu().numpy().ravel() for g in grads])


def load_predictor(model, args, seed):
    run_dir = (
        Path(args.checkpoint_root)
        / args.predictor_source
        / f"lambda_{args.risk_aversion:g}"
        / f"seed_{seed}"
    )
    path = run_dir / "best_model.pt"

    if not path.exists():
        raise FileNotFoundError(f"학습된 체크포인트가 없습니다: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]

    prefix = "return_model."
    if any(key.startswith(prefix) for key in state):
        state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }

    model.return_model.load_state_dict(state, strict=True)
    model.eval()
    return path


def evaluate(model, features, targets, covariance, args):
    predicted, weights, risk_factor = model(
        features=features,
        covariance=covariance,
    )

    # Oracle cost는 고정된 Predictor에 대한 Gradient가 0이므로 생략.
    # 따라서 아래 cost의 Gradient는 Regret Gradient와 동일하다.
    cost = markowitz_cost(
        weights, targets.to(torch.float64), risk_factor
    ).mean()

    mse = F.mse_loss(predicted, targets)
    combined = (
        args.alpha * cost
        + (1 - args.alpha) * args.mse_scale * mse.to(torch.float64)
    )

    params = tuple(model.return_model.parameters())

    grad_mu = torch.autograd.grad(
        cost, predicted, retain_graph=True
    )[0]

    grad_regret = torch.autograd.grad(
        cost, params, retain_graph=True
    )

    grad_total = torch.autograd.grad(combined, params)

    return {
        "prediction": predicted.detach().cpu().numpy(),
        "weights": weights.detach().cpu().numpy(),
        "gradient_mu": grad_mu.detach().cpu().numpy().ravel(),
        "gradient_regret": flatten(grad_regret),
        "gradient_total": flatten(grad_total),
    }


def compare(before, after, method, seed, quantile, date, covariance, delta, args):
    row = {
        "model": method,
        "seed": seed,
        "target_quantile": quantile,
        "center_date": date,
        "relative_covariance_change": norm(delta) / max(norm(covariance), 1e-30),
        "condition_median": float(np.median(np.linalg.cond(covariance))),
        "max_abs_prediction_change": float(
            np.max(np.abs(after["prediction"] - before["prediction"]))
        ),
        "max_abs_weight_change": float(
            np.max(np.abs(after["weights"] - before["weights"]))
        ),
    }

    for key in ["gradient_mu", "gradient_regret", "gradient_total"]:
        u, v = before[key], after[key]
        base_norm = norm(u)
        row[f"{key}_before_norm"] = base_norm
        row[f"{key}_abs_change"] = norm(v - u)
        row[f"{key}_rel_change"] = (
            norm(v - u) / base_norm if base_norm > 1e-12 else float("nan")
        )
        row[f"{key}_cosine"] = similarity(u, v)

    for tag, w in [("before", before["weights"]), ("after", after["weights"])]:
        row[f"active_lower_{tag}"] = int((w <= args.active_tolerance).sum())
        row[f"active_upper_{tag}"] = int(
            (w >= args.max_weight - args.active_tolerance).sum()
        )

    row["active_lower_changes"] = int(np.count_nonzero(
        (before["weights"] <= args.active_tolerance)
        != (after["weights"] <= args.active_tolerance)
    ))
    row["active_upper_changes"] = int(np.count_nonzero(
        (before["weights"] >= args.max_weight - args.active_tolerance)
        != (after["weights"] >= args.max_weight - args.active_tolerance)
    ))

    return row


def main():
    args = parse_args()

    if args.eps <= 0 or not 0 <= args.rho <= 1 or args.batch_size < 1:
        raise ValueError("eps > 0, 0 <= rho <= 1, batch_size >= 1이어야 합니다.")

    if not all(0 <= q <= 1 for q in args.quantiles):
        raise ValueError("quantile은 0~1 사이여야 합니다.")

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    common = dict(
        price_csv=args.price_csv,
        lookback=args.lookback,
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

    sample = RollingMVODataset(
        **common,
        risk_cache_dir="implementations/dfl_mvo_lee2025/cache/risk_matrix",
    )
    ridge = CovarianceBaselineDataset(
        **common,
        method="ridge",
        ridge_shrinkage=args.rho,
    )

    if (
        sample.tickers != ridge.tickers
        or not sample.target_dates.equals(ridge.target_dates)
    ):
        raise RuntimeError("Sample과 Ridge의 종목 또는 날짜가 다릅니다.")

    train_subset, _, _ = chronological_split(
        sample, args.train_end, args.validation_end
    )
    train_indices = np.array(train_subset.indices, dtype=int)

    if len(train_indices) < args.batch_size:
        raise RuntimeError("Train Window 수가 Batch Size보다 작습니다.")

    mean, std = fit_feature_standardizer(sample, train_subset)
    s = sample.covariances.detach().cpu().numpy()
    r = ridge.covariances.detach().cpu().numpy()

    eigen = np.linalg.eigvalsh(s[train_indices])
    cond = eigen[:, -1] / eigen[:, 0]

    # 동일한 PSD rank-one 섭동 방향을 모든 방법과 Seed에 사용한다.
    rng = np.random.default_rng(args.direction_seed)
    v = rng.normal(size=sample.n_assets)
    v /= np.linalg.norm(v)
    projector = np.outer(v, v)

    batches = []

    for q in args.quantiles:
        loc = int(np.argmin(np.abs(cond - np.quantile(cond, q))))
        start = int(np.clip(
            loc - args.batch_size // 2,
            0,
            len(train_indices) - args.batch_size,
        ))
        ix = train_indices[start:start + args.batch_size]

        raw = torch.stack([sample[int(i)]["features"] for i in ix])
        features = ((raw - mean) / std).to(torch.float32)
        targets = torch.stack(
            [sample[int(i)]["target"] for i in ix]
        ).to(torch.float32)

        # ||Delta_t||_F = eps * ||Sigma_sample,t||_F
        # Sample과 Ridge에 동일한 Delta_t를 더한다.
        delta = (
            np.linalg.norm(s[ix], axis=(1, 2))[:, None, None]
            * args.eps
            * projector[None]
        )

        date = str(sample.target_dates[ix[len(ix) // 2]].date())
        batches.append((
            q, date, ix, features, targets, torch.from_numpy(delta)
        ))

    rows = []

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = MLPWithMarkowitz(
            n_assets=sample.n_assets,
            lookback=args.lookback,
            hidden_dim=args.hidden_dim,
            dropout=0.0,
            risk_aversion=args.risk_aversion,
            max_weight=args.max_weight,
        )

        checkpoint = load_predictor(model, args, seed)
        print(f"[SEED {seed}] {checkpoint}", flush=True)

        for q, date, ix, features, targets, delta in batches:
            for name, base in [("Sample", s), ("Ridge", r)]:
                x = torch.from_numpy(np.ascontiguousarray(base[ix]))

                before = evaluate(model, features, targets, x, args)
                after = evaluate(model, features, targets, x + delta, args)

                row = compare(
                    before, after, name, seed, q, date,
                    x.numpy(), delta.numpy(), args,
                )

                # 섭동 없이 같은 입력을 다시 계산하여 수치적 잡음 확인
                if not args.skip_aa:
                    aa = evaluate(model, features, targets, x, args)
                    row["aa_regret_grad_abs_change"] = norm(
                        aa["gradient_regret"] - before["gradient_regret"]
                    )
                    row["aa_total_grad_abs_change"] = norm(
                        aa["gradient_total"] - before["gradient_total"]
                    )
                    row["aa_weight_max_change"] = float(np.max(
                        np.abs(aa["weights"] - before["weights"])
                    ))

                rows.append(row)

                print(
                    f"  {name:6} q={q:.2f} "
                    f"grad={row['gradient_regret_rel_change']:.3e} "
                    f"cos={row['gradient_regret_cosine']:.6f} "
                    f"wmax={row['max_abs_weight_change']:.3e}",
                    flush=True,
                )

    results = pd.DataFrame(rows)

    output = (
        Path(args.out_dir)
        / args.predictor_source
        / f"lambda_{args.risk_aversion:g}"
    )
    output.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output / "gradient_perturbation_per_case.csv",
        index=False,
    )

    summary = (
        results.groupby(["model", "target_quantile"], as_index=False)
        .agg(
            n=("seed", "size"),
            rel_grad_regret_median=("gradient_regret_rel_change", "median"),
            rel_grad_regret_p90=(
                "gradient_regret_rel_change",
                lambda x: x.quantile(0.9),
            ),
            rel_grad_total_median=("gradient_total_rel_change", "median"),
            rel_grad_mu_median=("gradient_mu_rel_change", "median"),
            weight_max_change_median=("max_abs_weight_change", "median"),
            active_upper_changes_total=("active_upper_changes", "sum"),
            active_lower_changes_total=("active_lower_changes", "sum"),
        )
    )

    summary.to_csv(
        output / "gradient_perturbation_summary.csv",
        index=False,
    )

    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print("\n" + summary.to_string(index=False))
    print(f"\n[DONE] {output.resolve()}")
    print("고정된 Predictor에서는 예측 수익률 변화량이 0이어야 합니다.")


if __name__ == "__main__":
    main()
