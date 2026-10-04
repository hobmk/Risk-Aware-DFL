import torch

from implementations.shrinkage_rcr_dfl.src.shrinkage_risk import (
    build_shrinkage_risk,
)


def _residuals():
    torch.manual_seed(123)

    return torch.randn(
        4,
        60,
        10,
        dtype=torch.float64,
    )


def test_shrinkage_is_symmetric_psd():
    result = build_shrinkage_risk(
        residuals=_residuals(),
        shrinkage_c=0.5,
        scale_mode="none",
    )

    matrix = result.risk_matrix

    assert torch.allclose(
        matrix,
        matrix.transpose(-1, -2),
        atol=1e-10,
    )

    eigenvalues = torch.linalg.eigvalsh(
        matrix
    )

    assert eigenvalues.min() >= -1e-10


def test_shrinkage_factors_are_between_zero_and_one():
    result = build_shrinkage_risk(
        residuals=_residuals(),
        shrinkage_c=0.5,
        scale_mode="none",
    )

    assert (
        result.shrinkage_factors >= 0
    ).all()

    assert (
        result.shrinkage_factors <= 1
    ).all()


def test_larger_c_means_stronger_shrinkage():
    residuals = _residuals()

    weak = build_shrinkage_risk(
        residuals=residuals,
        shrinkage_c=0.25,
        scale_mode="none",
    )

    strong = build_shrinkage_risk(
        residuals=residuals,
        shrinkage_c=2.0,
        scale_mode="none",
    )

    assert (
        strong.retained_ratio
        <= weak.retained_ratio + 1e-12
    ).all()


def test_c_zero_recovers_full_residual_covariance():
    result = build_shrinkage_risk(
        residuals=_residuals(),
        shrinkage_c=0.0,
        scale_mode="none",
    )

    assert torch.allclose(
        result.risk_matrix,
        result.residual_covariance,
        atol=0.0,
        rtol=0.0,
    )


def test_preserve_residual_trace():
    result = build_shrinkage_risk(
        residuals=_residuals(),
        shrinkage_c=1.0,
        scale_mode="preserve_residual_trace",
    )

    residual_trace = torch.diagonal(
        result.residual_covariance,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    risk_trace = torch.diagonal(
        result.risk_matrix,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    assert torch.allclose(
        residual_trace,
        risk_trace,
        atol=1e-10,
        rtol=1e-8,
    )