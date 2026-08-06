import torch

from implementations.rcr_dfl.src.residual_risk import (
    correlation_matrix,
    covariance_matrix,
    matrix_diagnostics,
    scale_correlation_to_covariance,
    shrink_correlation,
)


def test_correlation_is_symmetric_with_unit_diagonal() -> None:
    torch.manual_seed(3)
    values = torch.randn(80, 4, dtype=torch.float64)
    correlation = correlation_matrix(values)
    assert torch.allclose(correlation, correlation.T, atol=1e-12)
    assert torch.allclose(torch.diagonal(correlation), torch.ones(4, dtype=torch.float64), atol=1e-12)


def test_shrinkage_moves_off_diagonal_toward_zero() -> None:
    correlation = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
    shrunk = shrink_correlation(correlation, shrinkage=0.25)
    expected = torch.tensor([[1.0, 0.6], [0.6, 1.0]], dtype=torch.float64)
    assert torch.allclose(shrunk, expected, atol=1e-12)


def test_trace_scaling_matches_reference_trace() -> None:
    torch.manual_seed(5)
    values = torch.randn(100, 3, dtype=torch.float64) * 0.02
    covariance = covariance_matrix(values)
    correlation = correlation_matrix(values)
    a_res = scale_correlation_to_covariance(correlation, covariance, scaling="trace")
    assert torch.allclose(torch.trace(a_res), torch.trace(covariance), atol=1e-12)


def test_matrix_diagnostics_condition_number() -> None:
    matrix = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    diagnostics = matrix_diagnostics(matrix)
    assert diagnostics.symmetry_error.item() == 0.0
    assert diagnostics.minimum_eigenvalue.item() == 1.0
    assert diagnostics.maximum_eigenvalue.item() == 4.0
    assert diagnostics.condition_number.item() == 4.0
