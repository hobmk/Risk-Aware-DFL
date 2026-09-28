import torch

from implementations.eigen_rcr_dfl.src.capm import fit_capm
from implementations.eigen_rcr_dfl.src.eigen_risk import (
    build_eigenmode_risk,
    covariance_matrix,
)
from implementations.eigen_rcr_dfl.src.risk_pipeline import (
    build_eigen_rcr,
)


DTYPE = torch.float64


def make_data():
    torch.manual_seed(42)

    returns = torch.randn(
        60,
        30,
        dtype=DTYPE,
    ) * 0.01

    market = returns.mean(dim=1)

    return returns, market


def test_capm_shape():
    returns, market = make_data()

    result = fit_capm(
        returns,
        market,
    )

    assert result.alpha.shape == (30,)
    assert result.beta.shape == (30,)
    assert result.residuals.shape == (60, 30)


def test_eigen_risk_is_symmetric_psd():
    returns, market = make_data()

    residuals = fit_capm(
        returns,
        market,
    ).residuals

    result = build_eigenmode_risk(
        residuals,
        k=5,
    )

    assert torch.allclose(
        result.risk_matrix,
        result.risk_matrix.T,
        atol=1e-12,
    )

    eigenvalues = torch.linalg.eigvalsh(
        result.risk_matrix
    )

    assert eigenvalues.min() >= -1e-10


def test_k_n_recovers_residual_covariance():
    returns, market = make_data()

    residuals = fit_capm(
        returns,
        market,
    ).residuals

    full_covariance = covariance_matrix(
        residuals
    )

    result = build_eigenmode_risk(
        residuals,
        k=30,
        scale_mode="none",
    )

    assert torch.allclose(
        result.risk_matrix,
        full_covariance,
        atol=1e-10,
        rtol=1e-8,
    )


def test_eta_zero_recovers_base_covariance():
    returns, market = make_data()

    result = build_eigen_rcr(
        asset_returns=returns,
        market_returns=market,
        eigen_k=5,
        eta=0.0,
    )

    assert torch.allclose(
        result.effective_covariance,
        result.base_covariance,
        atol=1e-12,
    )


def test_explained_variance():
    returns, market = make_data()

    residuals = fit_capm(
        returns,
        market,
    ).residuals

    result = build_eigenmode_risk(
        residuals,
        k=5,
    )

    assert 0.0 <= result.retained_ratio <= 1.0

    assert torch.all(
        result.cumulative_explained_ratio[1:]
        >= result.cumulative_explained_ratio[:-1]
    )