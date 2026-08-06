import torch

from implementations.rcr_dfl.src.capm import fit_capm


def test_fit_capm_recovers_parameters() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    market = torch.randn(200, dtype=dtype) * 0.01
    alpha_true = torch.tensor([0.0002, -0.0001, 0.0004], dtype=dtype)
    beta_true = torch.tensor([0.7, 1.1, 1.4], dtype=dtype)
    assets = alpha_true + market.unsqueeze(-1) * beta_true + torch.randn(200, 3, dtype=dtype) * 0.0001
    result = fit_capm(assets, market)
    assert torch.allclose(result.alpha, alpha_true, atol=3e-5)
    assert torch.allclose(result.beta, beta_true, atol=2e-2)
    assert result.residuals.mean(dim=0).abs().max() < 1e-12


def test_capm_residual_is_orthogonal_to_market() -> None:
    torch.manual_seed(11)
    market = torch.randn(120, dtype=torch.float64) * 0.01
    assets = market.unsqueeze(-1) * torch.tensor([0.8, 1.2], dtype=torch.float64)
    assets += torch.randn(120, 2, dtype=torch.float64) * 0.001
    residuals = fit_capm(assets, market).residuals
    covariance = ((market - market.mean()).unsqueeze(-1) * residuals).sum(dim=0)
    assert torch.allclose(covariance, torch.zeros_like(covariance), atol=1e-12)


def test_fit_capm_supports_batch() -> None:
    torch.manual_seed(13)
    market = torch.randn(4, 60, dtype=torch.float64) * 0.01
    beta = torch.tensor([0.9, 1.1, 1.3], dtype=torch.float64)
    assets = market.unsqueeze(-1) * beta + torch.randn(4, 60, 3, dtype=torch.float64) * 0.001
    result = fit_capm(assets, market)
    assert result.alpha.shape == (4, 3)
    assert result.beta.shape == (4, 3)
    assert result.residuals.shape == (4, 60, 3)
