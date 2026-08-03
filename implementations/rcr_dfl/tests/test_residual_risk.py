import torch

from implementations.rcr_dfl.src.residual_risk import (
    correlation_matrix,
    covariance_matrix,
    normalize_covariance_trace,
)


def test_covariance_matches_torch_cov() -> None:
    torch.manual_seed(17)
    values = torch.randn(80, 5, dtype=torch.float64)

    actual = covariance_matrix(values)
    expected = torch.cov(values.T)

    assert torch.allclose(actual, expected, atol=1e-12)


def test_correlation_is_symmetric_with_unit_diagonal() -> None:
    torch.manual_seed(19)
    values = torch.randn(60, 4, dtype=torch.float64)
    correlation = correlation_matrix(values)

    assert torch.allclose(correlation, correlation.T, atol=1e-12)
    assert torch.allclose(
        torch.diagonal(correlation),
        torch.ones(4, dtype=torch.float64),
        atol=1e-12,
    )


def test_trace_normalization_matches_reference_trace() -> None:
    source = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
    reference = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))
    normalized = normalize_covariance_trace(source, reference)

    assert torch.allclose(torch.trace(normalized), torch.trace(reference), atol=1e-12)
