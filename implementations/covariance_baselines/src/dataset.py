from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from implementations.dfl_mvo_lee2025.src.dataset import RollingMVODataset


CovarianceMethod = Literal["ridge", "ledoit_wolf"]


def _sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())

    return digest.hexdigest()


def _cache_key(config: dict) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(
        payload
    ).hexdigest()[:20]


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + f".{os.getpid()}.tmp"
    )

    torch.save(
        payload,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def sample_covariance(
    window: np.ndarray,
) -> np.ndarray:
    covariance = np.cov(
        window,
        rowvar=False,
        ddof=1,
    )

    return 0.5 * (
        covariance
        + covariance.T
    )


def ridge_covariance(
    window: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(
            "ridge shrinkage는 0 이상 1 이하여야 합니다."
        )

    covariance = sample_covariance(
        window
    )

    n_assets = covariance.shape[0]

    mean_variance = (
        np.trace(covariance)
        / n_assets
    )

    target = (
        mean_variance
        * np.eye(
            n_assets,
            dtype=np.float64,
        )
    )

    covariance = (
        (1.0 - shrinkage)
        * covariance
        + shrinkage
        * target
    )

    return 0.5 * (
        covariance
        + covariance.T
    )


def ledoit_wolf_covariance(
    window: np.ndarray,
) -> tuple[np.ndarray, float]:
    estimator = LedoitWolf(
        assume_centered=False,
    )

    estimator.fit(window)

    covariance = np.asarray(
        estimator.covariance_,
        dtype=np.float64,
    )

    # sklearn LedoitWolf는 1/T covariance scale을 사용하므로
    # 기존 DFL-MVO/Ridge의 ddof=1, 즉 1/(T-1) scale과 맞춘다.
    n_samples = window.shape[0]

    if n_samples > 1:
        covariance = (
            covariance
            * n_samples
            / (n_samples - 1)
        )

    covariance = 0.5 * (
        covariance
        + covariance.T
    )

    return (
        covariance,
        float(estimator.shrinkage_),
    )

class CovarianceBaselineDataset(
    RollingMVODataset
):
    def __init__(
        self,
        price_csv: str | Path,
        method: CovarianceMethod,
        ridge_shrinkage: float = 0.10,
        lookback: int = 60,
        date_column: str = "Date",
        return_type: str = "simple",
        covariance_jitter: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        risk_cache_dir: str | Path = (
            "implementations/"
            "covariance_baselines/"
            "cache/risk_matrix"
        ),
        force_rebuild_risk_cache: bool = False,
    ) -> None:
        if method not in {
            "ridge",
            "ledoit_wolf",
        }:
            raise ValueError(
                "method는 ridge 또는 ledoit_wolf여야 합니다."
            )

        self.method = method
        self.ridge_shrinkage = float(
            ridge_shrinkage
        )

        super().__init__(
            price_csv=price_csv,
            lookback=lookback,
            date_column=date_column,
            return_type=return_type,
            covariance_jitter=covariance_jitter,
            dtype=dtype,
            risk_cache_dir=None,
        )

        self.risk_cache_dir = Path(
            risk_cache_dir
        )

        self.force_rebuild_risk_cache = bool(
            force_rebuild_risk_cache
        )

        self.risk_cache_hit = False
        self.risk_cache_key = None
        self.risk_cache_path = None

        self.ledoit_wolf_shrinkages = None

        self._load_or_build_risk_cache()

    def _cache_config(
        self,
    ) -> dict:
        return {
            "cache_version": (
                "covariance_baseline_v2"
            ),
            "method": self.method,
            "ridge_shrinkage": (
                self.ridge_shrinkage
                if self.method == "ridge"
                else None
            ),
            "lookback": self.lookback,
            "covariance_jitter": (
                self.covariance_jitter
            ),
            "dtype": str(
                self.dtype
            ),
            "n_dates": int(
                self.returns.shape[0]
            ),
            "n_assets": int(
                self.n_assets
            ),
            "returns_sha256": (
                _sha256_array(
                    self.returns
                )
            ),
        }

    def _load_or_build_risk_cache(
        self,
    ) -> None:
        config = self._cache_config()

        key = _cache_key(
            config
        )

        path = (
            self.risk_cache_dir
            / f"{self.method}_{key}.pt"
        )

        self.risk_cache_key = key
        self.risk_cache_path = str(
            path
        )

        if (
            path.exists()
            and not self.force_rebuild_risk_cache
        ):
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            self.covariances = payload[
                "covariances"
            ]

            self.ledoit_wolf_shrinkages = (
                payload.get(
                    "ledoit_wolf_shrinkages"
                )
            )

            self.risk_cache_hit = True

            print(
                f"[RiskCache] HIT: {path}"
            )

            return

        print(
            f"[RiskCache] BUILD: {path}"
        )

        n_samples = len(
            self.target_positions
        )

        covariances = torch.empty(
            (
                n_samples,
                self.n_assets,
                self.n_assets,
            ),
            dtype=self.dtype,
        )

        shrinkages = []

        eye = np.eye(
            self.n_assets,
            dtype=np.float64,
        )

        for sample_index, target_position in enumerate(
            self.target_positions
        ):
            target_position = int(
                target_position
            )

            window = self.returns[
                target_position - self.lookback:
                target_position
            ]

            if self.method == "ridge":
                covariance = (
                    ridge_covariance(
                        window,
                        self.ridge_shrinkage,
                    )
                )

            else:
                (
                    covariance,
                    shrinkage,
                ) = ledoit_wolf_covariance(
                    window
                )

                shrinkages.append(
                    shrinkage
                )

            covariance = (
                covariance
                + self.covariance_jitter
                * eye
            )

            covariance = 0.5 * (
                covariance
                + covariance.T
            )

            covariances[
                sample_index
            ] = torch.tensor(
                covariance,
                dtype=self.dtype,
            )

        self.covariances = (
            covariances.contiguous()
        )

        if self.method == "ledoit_wolf":
            self.ledoit_wolf_shrinkages = torch.tensor(
                shrinkages,
                dtype=torch.float64,
            )

        _atomic_save(
            {
                "config": config,
                "covariances": (
                    self.covariances
                ),
                "ledoit_wolf_shrinkages": (
                    self.ledoit_wolf_shrinkages
                ),
            },
            path,
        )

        print(
            f"[RiskCache] SAVED: {path}"
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        sample = dict(
            super().__getitem__(
                index
            )
        )

        sample[
            "sample_index"
        ] = index

        sample[
            "eigen_risk"
        ] = torch.zeros_like(
            sample["covariance"]
        )

        return sample