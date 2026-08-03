import torch

from implementations.rcr_dfl.src.effective_covariance import (
    build_effective_covariance,
    project_to_psd,
)


def test_eta_zero_returns_original_covariance() -> None:
    covariance = torch.tensor(
        [[0.04, 0.01], [0.01, 0.09]],
        dtype=torch.float64,
    )
    residual = torch.tensor(
        [[0.02, 0.005], [0.005, 0.03]],
        dtype=torch.float64,
    )

    effective = build_effective_covariance(
        covariance,
        residual,
        eta=0.0,
        normalization="trace",
    )

    assert torch.allclose(effective, covariance, atol=1e-12)


def test_effective_covariance_is_positive_definite_with_jitter() -> None:
    covariance = torch.tensor(
        [[0.04, 0.01], [0.01, 0.09]],
        dtype=torch.float64,
    )
    residual = torch.tensor(
        [[0.02, 0.015], [0.015, 0.03]],
        dtype=torch.float64,
    )

    effective = build_effective_covariance(
        covariance,
        residual,
        eta=0.5,
        normalization="trace",
        jitter=1e-8,
    )
    eigenvalues = torch.linalg.eigvalsh(effective)

    assert torch.all(eigenvalues > 0)


def test_project_to_psd_clips_negative_eigenvalues() -> None:
    matrix = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0]],
        dtype=torch.float64,
    )
    projected = project_to_psd(matrix, minimum_eigenvalue=1e-8)

    assert torch.linalg.eigvalsh(projected).min() >= 0



def test_eta_zero_accepts_zero_residual_covariance() -> None:
    covariance = torch.tensor(
        [[0.04, 0.01], [0.01, 0.09]],
        dtype=torch.float64,
    )
    zero_residual = torch.zeros_like(covariance)

    effective = build_effective_covariance(
        covariance,
        zero_residual,
        eta=0.0,
        normalization="trace",
    )

    assert torch.allclose(effective, covariance, atol=1e-12)
