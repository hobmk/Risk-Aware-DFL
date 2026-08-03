import torch

from implementations.rcr_dfl.src.capm import fit_capm


def test_fit_capm_recovers_parameters() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    time_steps = 200

    market = torch.randn(time_steps, dtype=dtype) * 0.01
    alpha_true = torch.tensor([0.0002, -0.0001, 0.0004], dtype=dtype)
    beta_true = torch.tensor([0.7, 1.1, 1.4], dtype=dtype)
    noise = torch.randn(time_steps, 3, dtype=dtype) * 0.0001
    assets = alpha_true + market.unsqueeze(-1) * beta_true + noise

    result = fit_capm(assets, market)

    assert result.residuals.shape == assets.shape
    assert torch.allclose(result.alpha, alpha_true, atol=3e-5)
    assert torch.allclose(result.beta, beta_true, atol=2e-2)
    assert torch.allclose(result.residuals.mean(dim=0), torch.zeros(3, dtype=dtype), atol=1e-12)


def test_capm_residual_is_orthogonal_to_market() -> None:
    torch.manual_seed(11)
    dtype = torch.float64
    market = torch.randn(120, dtype=dtype) * 0.01
    assets = market.unsqueeze(-1) * torch.tensor([0.8, 1.2], dtype=dtype)
    assets = assets + torch.randn(120, 2, dtype=dtype) * 0.001

    result = fit_capm(assets, market)
    centered_market = market - market.mean()
    covariance = (centered_market.unsqueeze(-1) * result.residuals).sum(dim=0)

    assert torch.allclose(covariance, torch.zeros_like(covariance), atol=1e-12)


def test_fit_capm_supports_batch() -> None:
    torch.manual_seed(13)
    dtype = torch.float64
    market = torch.randn(4, 60, dtype=dtype) * 0.01
    beta = torch.tensor([0.9, 1.1, 1.3], dtype=dtype)
    assets = market.unsqueeze(-1) * beta + torch.randn(4, 60, 3, dtype=dtype) * 0.001

    result = fit_capm(assets, market)

    assert result.alpha.shape == (4, 3)
    assert result.beta.shape == (4, 3)
    assert result.residuals.shape == (4, 60, 3)
