import torch

from implementations.eigen_rcr_dfl.src.capm import fit_capm
from implementations.eigen_rcr_dfl.src.eigen_risk import covariance_matrix
from implementations.decomposition_rcr_dfl.src.decomposition_risk import (
    build_decomposition_risk,
)


def _data():
    torch.manual_seed(123)

    market = torch.randn(
        4,
        60,
        dtype=torch.float64,
    )

    beta = torch.linspace(
        0.5,
        1.5,
        10,
        dtype=torch.float64,
    )

    noise = (
        0.3
        * torch.randn(
            4,
            60,
            10,
            dtype=torch.float64,
        )
    )

    assets = (
        market.unsqueeze(-1)
        * beta
        + noise
    )

    capm = fit_capm(
        asset_returns=assets,
        market_returns=market,
        fit_intercept=True,
    )

    return assets, market, capm


def test_k_zero_preserves_residual_diagonal():
    assets, market, capm = _data()

    result = build_decomposition_risk(
        asset_returns=assets,
        market_returns=market,
        beta=capm.beta,
        residuals=capm.residuals,
        k=0,
        covariance_jitter=0.0,
    )

    reconstructed_diag = torch.diagonal(
        result.reconstructed_residual,
        dim1=-2,
        dim2=-1,
    )

    residual_diag = torch.diagonal(
        result.residual_covariance,
        dim1=-2,
        dim2=-1,
    )

    assert torch.allclose(
        reconstructed_diag,
        residual_diag,
        atol=1e-10,
        rtol=1e-8,
    )


def test_any_k_preserves_residual_diagonal():
    assets, market, capm = _data()

    result = build_decomposition_risk(
        asset_returns=assets,
        market_returns=market,
        beta=capm.beta,
        residuals=capm.residuals,
        k=3,
        covariance_jitter=0.0,
    )

    assert torch.allclose(
        torch.diagonal(
            result.reconstructed_residual,
            dim1=-2,
            dim2=-1,
        ),
        torch.diagonal(
            result.residual_covariance,
            dim1=-2,
            dim2=-1,
        ),
        atol=1e-10,
        rtol=1e-8,
    )


def test_k_n_recovers_full_residual_covariance():
    assets, market, capm = _data()

    n_assets = assets.shape[-1]

    result = build_decomposition_risk(
        asset_returns=assets,
        market_returns=market,
        beta=capm.beta,
        residuals=capm.residuals,
        k=n_assets,
        covariance_jitter=0.0,
    )

    assert torch.allclose(
        result.reconstructed_residual,
        result.residual_covariance,
        atol=0.0,
        rtol=0.0,
    )


def test_k_n_recovers_base_covariance():
    assets, market, capm = _data()

    result = build_decomposition_risk(
        asset_returns=assets,
        market_returns=market,
        beta=capm.beta,
        residuals=capm.residuals,
        k=assets.shape[-1],
        covariance_jitter=0.0,
    )

    base = covariance_matrix(
        assets
    )

    assert torch.allclose(
        result.effective_covariance,
        base,
        atol=1e-10,
        rtol=1e-8,
    )


def test_effective_covariance_is_symmetric_psd():
    assets, market, capm = _data()

    result = build_decomposition_risk(
        asset_returns=assets,
        market_returns=market,
        beta=capm.beta,
        residuals=capm.residuals,
        k=3,
        covariance_jitter=1e-6,
    )

    matrix = result.effective_covariance

    assert torch.allclose(
        matrix,
        matrix.transpose(-1, -2),
        atol=1e-10,
    )

    eigenvalues = torch.linalg.eigvalsh(
        matrix
    )

    assert eigenvalues.min() >= -1e-10