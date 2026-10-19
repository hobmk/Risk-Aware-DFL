from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)

    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())

    return digest.hexdigest()


def _cache_key(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()[:20]


def _atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_suffix(
        path.suffix
        + f".{os.getpid()}.tmp"
    )

    torch.save(
        payload,
        tmp_path,
    )

    os.replace(
        tmp_path,
        path,
    )


def build_covariance_cache_config(
    *,
    returns: np.ndarray,
    lookback: int,
    covariance_jitter: float,
    dtype: torch.dtype,
) -> dict[str, Any]:

    return {
        "cache_version": "dfl_mvo_covariance_v1",
        "risk_method": "sample_covariance",
        "lookback": int(lookback),
        "covariance_jitter": float(
            covariance_jitter
        ),
        "dtype": str(dtype),
        "n_dates": int(
            returns.shape[0]
        ),
        "n_assets": int(
            returns.shape[1]
        ),
        "returns_sha256": _sha256_array(
            returns
        ),
    }


def load_or_build_covariance_cache(
    *,
    returns: np.ndarray,
    lookback: int,
    covariance_jitter: float,
    dtype: torch.dtype,
    cache_dir: str | Path,
    force_rebuild: bool = False,
) -> tuple[
    torch.Tensor,
    Path,
    bool,
    str,
]:

    config = (
        build_covariance_cache_config(
            returns=returns,
            lookback=lookback,
            covariance_jitter=(
                covariance_jitter
            ),
            dtype=dtype,
        )
    )

    key = _cache_key(
        config
    )

    path = (
        Path(cache_dir)
        / f"covariance_{key}.pt"
    )

    if (
        path.exists()
        and not force_rebuild
    ):
        payload = torch.load(
            path,
            map_location="cpu",
        )

        covariances = payload[
            "covariances"
        ]

        print(
            f"[RiskCache] HIT: "
            f"{path}"
        )

        return (
            covariances,
            path,
            True,
            key,
        )

    n_samples = (
        returns.shape[0]
        - lookback
    )

    n_assets = (
        returns.shape[1]
    )

    covariances = torch.empty(
        (
            n_samples,
            n_assets,
            n_assets,
        ),
        dtype=dtype,
    )

    eye = np.eye(
        n_assets,
        dtype=np.float64,
    )

    print(
        f"[RiskCache] BUILD: "
        f"{path}"
    )

    for (
        sample_index,
        target_position,
    ) in enumerate(
        range(
            lookback,
            returns.shape[0],
        )
    ):

        feature_window = returns[
            target_position - lookback:
            target_position
        ]

        covariance = np.cov(
            feature_window,
            rowvar=False,
            ddof=1,
        )

        covariance = 0.5 * (
            covariance
            + covariance.T
        )

        covariance = (
            covariance
            + covariance_jitter
            * eye
        )

        covariances[
            sample_index
        ] = torch.tensor(
            covariance,
            dtype=dtype,
        )

    _atomic_torch_save(
        {
            "config": config,
            "covariances": (
                covariances
            ),
        },
        path,
    )

    print(
        f"[RiskCache] SAVED: "
        f"{path}"
    )

    return (
        covariances,
        path,
        False,
        key,
    )


def build_oracle_cache_config(
    *,
    dataset,
    risk_aversion: float,
    max_weight: float,
) -> dict[str, Any]:

    return {
        "cache_version": (
            "dfl_mvo_oracle_v1"
        ),

        "risk_method": (
            "sample_covariance"
        ),

        "risk_cache_key": getattr(
            dataset,
            "risk_cache_key",
            None,
        ),

        "lookback": int(
            dataset.lookback
        ),

        "return_type": str(
            dataset.return_type
        ),

        "covariance_jitter": float(
            dataset.covariance_jitter
        ),

        "dtype": str(
            dataset.dtype
        ),

        "risk_aversion": float(
            risk_aversion
        ),

        "max_weight": float(
            max_weight
        ),

        "n_assets": int(
            dataset.n_assets
        ),

        "n_samples": int(
            len(dataset)
        ),

        "target_returns_sha256":
            _sha256_array(
                dataset.returns[
                    dataset.target_positions
                ]
            ),
    }


def load_or_build_oracle_cache(
    *,
    dataset,
    risk_aversion: float,
    max_weight: float,
    batch_size: int,
    cache_dir: str | Path,
    force_rebuild: bool = False,
) -> tuple[
    torch.Tensor,
    Path,
    bool,
]:

    if getattr(
        dataset,
        "covariances",
        None,
    ) is None:
        raise RuntimeError(
            "Oracle cache를 만들려면 "
            "covariance cache가 활성화되어 "
            "있어야 합니다."
        )

    config = (
        build_oracle_cache_config(
            dataset=dataset,
            risk_aversion=(
                risk_aversion
            ),
            max_weight=max_weight,
        )
    )

    key = _cache_key(
        config
    )

    path = (
        Path(cache_dir)
        / f"oracle_{key}.pt"
    )

    if (
        path.exists()
        and not force_rebuild
    ):
        payload = torch.load(
            path,
            map_location="cpu",
        )

        oracle_weights = payload[
            "oracle_weights"
        ]

        print(
            f"[OracleCache] HIT: "
            f"{path}"
        )

        return (
            oracle_weights,
            path,
            True,
        )

    print(
        f"[OracleCache] BUILD: "
        f"{path}"
    )

    markowitz_layer = (
        build_markowitz_layer(
            n_assets=dataset.n_assets,
            max_weight=max_weight,
        )
    )

    # 기존 코드와 동일하게
    # target을 먼저 dataset dtype(float32)으로
    # 만든 뒤 solver에서 float64로 변환한다.
    target_returns = torch.tensor(
        dataset.returns[
            dataset.target_positions
        ],
        dtype=dataset.dtype,
    ).to(
        dtype=torch.float64,
        device="cpu",
    )

    covariances = (
        dataset.covariances.to(
            dtype=torch.float64,
            device="cpu",
        )
    )

    batches = []

    n_batches = (
        len(dataset)
        + batch_size
        - 1
    ) // batch_size

    for (
        batch_index,
        start,
    ) in enumerate(
        range(
            0,
            len(dataset),
            batch_size,
        ),
        start=1,
    ):
        end = min(
            start + batch_size,
            len(dataset),
        )

        batch_targets = (
            target_returns[
                start:end
            ]
        )

        batch_covariances = (
            covariances[
                start:end
            ]
        )

        risk_factor = (
            covariance_to_risk_factor(
                covariance=(
                    batch_covariances
                ),
                risk_aversion=(
                    risk_aversion
                ),
                jitter=0.0,
            )
        )

        with torch.no_grad():
            batch_weights, = (
                markowitz_layer(
                    batch_targets,
                    risk_factor,
                )
            )

        batches.append(
            batch_weights
            .detach()
            .cpu()
            .to(torch.float64)
        )

        if (
            batch_index % 25 == 0
            or batch_index
            == n_batches
        ):
            print(
                f"[OracleCache] "
                f"{batch_index}/"
                f"{n_batches}"
            )

    oracle_weights = torch.cat(
        batches,
        dim=0,
    ).contiguous()

    _atomic_torch_save(
        {
            "config": config,
            "oracle_weights":
                oracle_weights,
        },
        path,
    )

    print(
        f"[OracleCache] SAVED: "
        f"{path}"
    )

    return (
        oracle_weights,
        path,
        False,
    )