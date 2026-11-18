
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t

MODELS = ("dfl_mvo", "ridge", "ledoit_wolf", "rank_tail", "topk")
ROOT = Path("implementations/final_benchmark/outputs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=ROOT / "predictor_quality")
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 25.0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    p.add_argument("--with-regret", action="store_true")
    return p.parse_args()


def validate_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame.index = pd.to_datetime(frame.index).normalize()
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"Duplicate dates/tickers in {path}")
    frame = frame.sort_index().sort_index(axis=1).astype(np.float64)
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Empty/nonfinite data in {path}")
    return frame


def read_long(path: Path, field: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "ticker", field}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing {required - set(df.columns)} in {path}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(f"Duplicate date/ticker in {path}")
    return validate_frame(df.pivot(index="date", columns="ticker", values=field), path)


def read_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Missing date in {path}")
    return validate_frame(df.set_index("date"), path)


def load_run(root: Path, model: str, lam: float, seed: int):
    path = root / model / f"lambda_{lam:g}" / f"seed_{seed}"
    if model == "dfl_mvo":
        file = path / "asset_predictions.csv"
        return read_long(file, "predicted_return"), read_long(file, "true_return")
    return read_wide(path / "predicted_returns.csv"), read_wide(path / "true_returns.csv")


def assert_aligned(a: pd.DataFrame, b: pd.DataFrame, name: str):
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        raise ValueError(f"Date/ticker mismatch: {name}")


def rank_ic_daily(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    x = rankdata(pred, method="average", axis=1)
    y = rankdata(truth, method="average", axis=1)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    numerator = np.sum(x * y, axis=1)
    return np.divide(numerator, denom, out=np.full(len(denom), np.nan), where=denom > 0)


def prepare_common_risk(price_csv: str, lookback: int, dates: pd.DatetimeIndex, tickers: pd.Index):
    from implementations.dfl_mvo_lee2025.src.dataset import RollingMVODataset
    import torch

    dataset = RollingMVODataset(
        price_csv=price_csv, lookback=lookback, return_type="simple",
        covariance_jitter=1e-6, risk_cache_dir=None, dtype=torch.float64,
    )
    ticker_to_col = {name: i for i, name in enumerate(dataset.tickers)}
    if set(tickers) != set(ticker_to_col):
        raise ValueError("Price dataset ticker universe != prediction ticker universe")
    order = [ticker_to_col[name] for name in tickers]
    position_map = dict(zip(pd.DatetimeIndex(dataset.target_dates).normalize(), dataset.target_positions))
    covs, targets = [], []

    for date in dates:
        if date not in position_map:
            raise ValueError(f"Test date not found in price dataset: {date}")
        pos = int(position_map[date])
        past = dataset.returns[pos - lookback:pos, :]
        cov = np.cov(past, rowvar=False, ddof=1)
        cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(len(dataset.tickers))
        covs.append(cov[np.ix_(order, order)])
        targets.append(dataset.returns[pos, order])

    return np.asarray(covs, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def solve_all(layer, mu: np.ndarray, factors, batch_size: int) -> np.ndarray:
    import torch

    result = []
    with torch.no_grad():
        for start in range(0, len(mu), batch_size):
            stop = min(start + batch_size, len(mu))
            mu_tensor = torch.as_tensor(mu[start:stop], dtype=torch.float64)
            weights, = layer(mu_tensor, factors[start:stop])
            result.append(weights.detach().cpu().numpy())
    weights = np.concatenate(result, axis=0)
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite MVO solver output")
    return weights


def evaluate_regret(weights, oracle, truth, covariance, lam):
    def cost(w):
        risk = np.einsum("ti,tij,tj->t", w, covariance, w)
        ret = np.einsum("ti,ti->t", w, truth)
        return lam * risk - ret

    daily = cost(weights) - cost(oracle)
    return {
        "common_cov_regret": float(daily.mean()),
        "common_cov_regret_min": float(daily.min()),
        "common_cov_negative_days": int(np.sum(daily < -1e-6)),
    }


def summarize(df: pd.DataFrame, metrics: list[str]):
    rows = []
    for (lam, model), sub in df.groupby(["lambda", "model"], sort=False):
        for metric in metrics:
            values = sub[metric].dropna().to_numpy()
            rows.append({"lambda": lam, "model": model, "metric": metric,
                         "n_seeds": len(values), "mean": values.mean(),
                         "std": values.std(ddof=1) if len(values) > 1 else np.nan})
    return pd.DataFrame(rows)


def paired_summary(df: pd.DataFrame, metrics: list[str]):
    ref = df[df.model == "dfl_mvo"][["lambda", "seed", *metrics]]
    paired = df[df.model != "dfl_mvo"].merge(ref, on=["lambda", "seed"],
                                              suffixes=("", "_dfl"), validate="many_to_one")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_dfl"]

    rows = []
    for (lam, model), sub in paired.groupby(["lambda", "model"], sort=False):
        for metric in metrics:
            values = sub[f"delta_{metric}"].dropna().to_numpy()
            n = len(values)
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            ci = float(t.ppf(0.975, n-1) * sem) if n > 1 else np.nan
            better = (values > 0) if metric == "rank_ic" else (values < 0)
            rows.append({"lambda": lam, "model": model, "metric": metric,
                         "n_seeds": n, "mean_delta": mean,
                         "ci95_low": mean-ci, "ci95_high": mean+ci,
                         "better_seeds": int(better.sum())})
    return paired, pd.DataFrame(rows)


def main():
    args = parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or args.batch_size < 1:
        raise ValueError("Invalid seeds or batch size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for lam in args.lambdas:
        dfl_pred, reference = load_run(args.root, "dfl_mvo", lam, args.seeds[0])
        assert_aligned(dfl_pred, reference, f"reference lambda={lam:g}")
        truth = reference.to_numpy()
        print(f"LAMBDA={lam:g} | dates={len(reference)} | assets={len(reference.columns)}", flush=True)
        factor, covs, oracle, layer = None, None, None, None

        if args.with_regret:
            import torch
            from implementations.dfl_mvo_lee2025.src.optimization import (
                build_markowitz_layer, covariance_to_risk_factor,
            )
            covs, source_truth = prepare_common_risk(
                args.price_csv, args.lookback, reference.index, reference.columns,
            )
            if not np.allclose(truth, source_truth, rtol=0, atol=1e-8):
                raise ValueError("CSV true returns do not match price dataset")
            factor = covariance_to_risk_factor(
                torch.as_tensor(covs, dtype=torch.float64), risk_aversion=lam,
            )
            layer = build_markowitz_layer(len(reference.columns), args.max_weight)
            oracle = solve_all(layer, truth, factor, args.batch_size)
            print(f"  Common covariance and oracle constructed for lambda={lam:g}", flush=True)

        for model in MODELS:
            for seed in args.seeds:
                pred_df, true_df = load_run(args.root, model, lam, seed)
                assert_aligned(pred_df, reference, f"{model} seed={seed} pred")
                assert_aligned(true_df, reference, f"{model} seed={seed} truth")
                pred = pred_df.to_numpy()
                max_diff = float(np.max(np.abs(true_df.to_numpy() - truth)))
                if max_diff > 1e-8:
                    raise ValueError(f"True return mismatch: {model} seed={seed}, max_diff={max_diff:.3e}")
                error = pred - truth
                ic = rank_ic_daily(pred, truth)
                if not np.isfinite(ic).any():
                    raise ValueError(f"No valid rank IC: {model} seed={seed}")
                row = {"lambda": lam, "model": model, "seed": seed,
                       "test_mse": float(np.mean(error ** 2)),
                       "test_mae": float(np.mean(np.abs(error))),
                       "rank_ic": float(np.nanmean(ic)),
                       "rank_ic_valid_days": int(np.isfinite(ic).sum())}
                if args.with_regret:
                    weights = solve_all(layer, pred, factor, args.batch_size)
                    row.update(evaluate_regret(weights, oracle, truth, covs, lam))
                    if model == "dfl_mvo" and seed == args.seeds[0]:
                        native_path = args.root / model / f"lambda_{lam:g}" / f"seed_{seed}" / "asset_predictions.csv"
                        saved_w = read_long(native_path, "predicted_weight").to_numpy()
                        saved_oracle = read_long(native_path, "oracle_weight").to_numpy()
                        replay_w_diff = np.max(np.abs(weights - saved_w))
                        replay_oracle_diff = np.max(np.abs(oracle - saved_oracle))
                        print(f"  DFL replay max weight diff={replay_w_diff:.3e}, "
                              f"oracle diff={replay_oracle_diff:.3e}", flush=True)
                        if max(replay_w_diff, replay_oracle_diff) > 5e-3:
                            raise RuntimeError("DFL replay mismatch > 5e-3: check solver and covariance")
                    if row["common_cov_regret_min"] < -1e-5:
                        print(f"WARNING: negative regret at {model} seed={seed}: "
                              f"{row['common_cov_regret_min']:.3e}", flush=True)
                rows.append(row)
                if args.with_regret:
                    pd.DataFrame(rows).to_csv(args.output_dir / "predictor_quality_partial.csv", index=False)
                print(f"OK | lambda={lam:g} | {model:12s} | seed={seed}", flush=True)

        pd.DataFrame(rows).to_csv(args.output_dir / "predictor_quality_partial.csv", index=False)

    runs = pd.DataFrame(rows)
    metrics = ["test_mse", "test_mae", "rank_ic"]
    if args.with_regret:
        metrics.append("common_cov_regret")
    summary = summarize(runs, metrics)
    paired, delta = paired_summary(runs, metrics)
    runs.to_csv(args.output_dir / "predictor_quality_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "predictor_quality_summary.csv", index=False)
    paired.to_csv(args.output_dir / "predictor_quality_paired.csv", index=False)
    delta.to_csv(args.output_dir / "predictor_quality_paired_summary.csv", index=False)
    print("\nMODEL SUMMARY\n", summary.to_string(index=False), flush=True)
    print("\nPAIRED DIFFERENCE (model - DFL-MVO)\n", delta.to_string(index=False), flush=True)
    print("\nSaved:", args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
