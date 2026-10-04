from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import EigenRCRDataset
from .model import EigenRCRDFL


def _cache_key(
    config: dict[str, Any],
) -> str:
    serialized = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:20]


def load_or_build_oracle_cache(
    model: EigenRCRDFL,
    dataset: EigenRCRDataset,
    batch_size: int,
    cache_dir: str | Path,
    cache_config: dict[str, Any],
    force_rebuild: bool = False,
) -> tuple[torch.Tensor, Path, bool]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = _cache_key(
        cache_config
    )

    cache_path = (
        cache_dir
        / f"oracle_{key}.pt"
    )

    if cache_path.exists() and not force_rebuild:
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        cached_config = payload.get(
            "config"
        )

        weights = payload.get(
            "weights"
        )

        valid = (
            cached_config == cache_config
            and isinstance(weights, torch.Tensor)
            and weights.shape
            == (
                len(dataset),
                dataset.n_assets,
            )
        )

        if valid:
            print(
                f"[OracleCache] HIT: "
                f"{cache_path}"
            )

            return (
                weights.to(
                    device="cpu",
                    dtype=torch.float64,
                ),
                cache_path,
                True,
            )

        print(
            "[OracleCache] Existing cache is invalid. "
            "Rebuilding."
        )

    print(
        f"[OracleCache] BUILD: "
        f"{cache_path}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    oracle_weights = torch.empty(
        (
            len(dataset),
            dataset.n_assets,
        ),
        dtype=torch.float64,
        device="cpu",
    )

    n_batches = len(dataloader)

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):
        sample_indices = batch[
            "sample_index"
        ].to(
            device="cpu",
            dtype=torch.long,
        )

        covariance = batch[
            "covariance"
        ].to(
            device="cpu",
            dtype=torch.float64,
        )

        eigen_risk = batch[
            "eigen_risk"
        ].to(
            device="cpu",
            dtype=torch.float64,
        )

        true_returns = batch[
            "target"
        ].to(
            device="cpu",
            dtype=torch.float64,
        )

        _, risk_factor = (
            model.build_risk_factor(
                covariance=covariance,
                eigen_risk=eigen_risk,
            )
        )

        batch_oracle_weights = (
            model.solve_oracle(
                true_returns=true_returns,
                risk_factor=risk_factor,
            )
        )

        oracle_weights.index_copy_(
            0,
            sample_indices,
            batch_oracle_weights.to(
                device="cpu",
                dtype=torch.float64,
            ),
        )

        if (
            batch_index % 25 == 0
            or batch_index == n_batches
        ):
            print(
                f"[OracleCache] "
                f"{batch_index}/{n_batches}"
            )

    torch.save(
        {
            "config": cache_config,
            "weights": oracle_weights,
        },
        cache_path,
    )

    print(
        f"[OracleCache] SAVED: "
        f"{cache_path}"
    )

    return (
        oracle_weights,
        cache_path,
        False,
    )