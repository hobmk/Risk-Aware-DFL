from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


RISK_CACHE_VERSION = "rank_tail_risk_v1"

CACHED_TENSOR_FIELDS = (
    "base_covariances",
    "covariances",
    "capm_alpha",
    "capm_beta",
    "market_covariances",
    "residual_covariances",
    "rank_tail_dependence",
    "rank_tail_risk",
    "tail_fraction",
    "mean_offdiag_dependence",
    "mean_abs_offdiag_dependence",
    "diagonal_preservation_error",
    "capm_decomposition_error",
    "minimum_effective_eigenvalue",
)


def _numpy_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_risk_cache_config(dataset) -> dict[str, Any]:
    return {
        "cache_version": RISK_CACHE_VERSION,
        "risk_method": "rank_tail_residual_dependence_v1",
        "lookback": int(dataset.lookback),
        "return_type": str(
            getattr(dataset, "return_type", "unknown")
        ),
        "covariance_jitter": float(
            dataset.covariance_jitter
        ),
        "risk_free_rate": float(
            dataset.risk_free_rate
        ),
        "fit_intercept": bool(
            dataset.fit_intercept
        ),
        "tail_q": float(dataset.tail_q),
        "tail_side": str(dataset.tail_side),
        "dtype": str(dataset.dtype),
        "n_assets": int(dataset.n_assets),
        "n_samples": int(
            len(dataset.returns) - dataset.lookback
        ),
        "returns_sha256": _numpy_sha256(
            dataset.returns
        ),
        "market_returns_sha256": _numpy_sha256(
            dataset.market_returns
        ),
    }


def risk_cache_key(
    config: dict[str, Any],
) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()[:20]


def risk_cache_path(
    cache_dir: str | Path,
    key: str,
) -> Path:
    return (
        Path(cache_dir)
        / f"rank_tail_risk_{key}.pt"
    )


def load_risk_cache(
    path: Path,
) -> dict[str, torch.Tensor]:
    payload = torch.load(
        path,
        map_location="cpu",
    )

    tensors = payload.get("tensors")

    if not isinstance(tensors, dict):
        raise ValueError(
            f"잘못된 Rank-Tail risk cache 형식입니다: "
            f"{path}"
        )

    missing = [
        name
        for name in CACHED_TENSOR_FIELDS
        if name not in tensors
    ]

    if missing:
        raise ValueError(
            f"Rank-Tail risk cache에 필드가 없습니다: "
            f"{missing} | {path}"
        )

    return tensors


def save_risk_cache(
    *,
    path: Path,
    config: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_suffix(
        path.suffix
        + f".{os.getpid()}.tmp"
    )

    cpu_tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in tensors.items()
    }

    torch.save(
        {
            "config": config,
            "tensors": cpu_tensors,
        },
        tmp_path,
    )

    os.replace(
        tmp_path,
        path,
    )