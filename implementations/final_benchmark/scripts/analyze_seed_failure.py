
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("implementations/final_benchmark/outputs")
data = pd.read_csv(root / "common_evaluation/all_runs_common_metrics.csv")

metrics = [
    "gross_sharpe",
    "gross_annualized_return",
    "gross_volatility",
    "gross_max_drawdown",
    "net_10bp_sharpe",
]
metrics = [m for m in metrics if m in data.columns]

data = data[data["model"].isin(["dfl_mvo", "rank_tail"])]
paired = data.pivot(index=["lambda", "seed"], columns="model", values=metrics).dropna()

result = pd.DataFrame(index=paired.index)

for metric in metrics:
    result[f"dfl_{metric}"] = paired[(metric, "dfl_mvo")]
    result[f"rt_{metric}"] = paired[(metric, "rank_tail")]
    result[f"delta_{metric}"] = result[f"rt_{metric}"] - result[f"dfl_{metric}"]

result["group"] = np.where(
    result["delta_gross_sharpe"] > 0, "improved", "worsened_or_tied"
)
result["return_sacrifice"] = (
    (result["delta_gross_annualized_return"] < 0)
    & (result["delta_gross_volatility"] < 0)
)

# Optional: previously computed Rank-Tail risk exposure
path = root / "mechanism/mechanism_daily.csv"

if path.exists():
    m = pd.read_csv(path)
    m = m[
        (m["mechanism"] == "rank_tail")
        & m["model"].isin(["dfl_mvo", "rank_tail"])
    ]
    ex = m.pivot_table(
        index=["lambda", "seed", "date"],
        columns="model",
        values="rt_positive_offdiag",
        aggfunc="mean",
    ).dropna()

    reduction = (ex["dfl_mvo"] - ex["rank_tail"]).groupby(
        level=["lambda", "seed"]
    ).mean()
    result["tail_exposure_reduction"] = reduction.reindex(result.index)

# Optional: previously computed future risk difference
path = root / "risk_reduction_future_risk/by_seed.csv"

if path.exists():
    f = pd.read_csv(path)
    f = f[
        (f["model"] == "rank_tail")
        & (f["horizon"] == 20)
        & (f["future_metric"] == "volatility")
    ]
    future = f.set_index(["lambda", "seed"])["mean_future_risk_reduction"]
    result["future_vol_reduction_h20"] = future.reindex(result.index)

result = result.reset_index()

out = root / "seed_failure_diagnosis"
out.mkdir(parents=True, exist_ok=True)
result.to_csv(out / "per_seed.csv", index=False)

summary = result.groupby(["lambda", "group"]).agg(
    seeds=("seed", "count"),
    mean_sharpe_gap=("delta_gross_sharpe", "mean"),
    median_sharpe_gap=("delta_gross_sharpe", "median"),
    mean_return_gap=("delta_gross_annualized_return", "mean"),
    mean_vol_gap=("delta_gross_volatility", "mean"),
    return_sacrifice_count=("return_sacrifice", "sum"),
)

for column in ["tail_exposure_reduction", "future_vol_reduction_h20"]:
    if column in result:
        summary[f"mean_{column}"] = result.groupby(
            ["lambda", "group"]
        )[column].mean()

summary.to_csv(out / "group_summary.csv")
print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
