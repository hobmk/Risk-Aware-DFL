import torch

from implementations.rcr_dfl.src.effective_covariance import build_effective_covariance, project_to_psd


def test_eta_zero_returns_original_covariance() -> None:
    covariance = torch.tensor([[0.04, 0.01], [0.01, 0.09]], dtype=torch.float64)
    a_res = torch.tensor([[0.05, 0.02], [0.02, 0.05]], dtype=torch.float64)
    effective = build_effective_covariance(covariance, a_res, eta=0.0)
    assert torch.allclose(effective, covariance, atol=1e-12)


def test_effective_covariance_matches_definition() -> None:
    covariance = torch.tensor([[0.04, 0.01], [0.01, 0.09]], dtype=torch.float64)
    a_res = torch.tensor([[0.05, 0.02], [0.02, 0.05]], dtype=torch.float64)
    effective = build_effective_covariance(covariance, a_res, eta=0.5)
    assert torch.allclose(effective, covariance + 0.5 * a_res, atol=1e-12)


def test_project_to_psd_clips_negative_eigenvalues() -> None:
    matrix = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    projected = project_to_psd(matrix, minimum_eigenvalue=1e-8)
    assert torch.linalg.eigvalsh(projected).min() >= 0
