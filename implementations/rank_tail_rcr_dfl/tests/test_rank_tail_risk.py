\
from __future__ import annotations

import torch

from implementations.rank_tail_rcr_dfl.src.rank_tail_risk import (
    build_rank_tail_residual_risk,
)


def test_rank_tail_shapes_symmetry_psd_and_diagonal():
    torch.manual_seed(7)

    residuals = torch.randn(
        4,
        60,
        6,
        dtype=torch.float64,
    )

    result = build_rank_tail_residual_risk(
        residuals=residuals,
        tail_q=0.20,
    )

    assert result.risk_matrix.shape == (
        4,
        6,
        6,
    )

    assert torch.allclose(
        result.risk_matrix,
        result.risk_matrix.transpose(
            -1,
            -2,
        ),
        atol=1e-12,
        rtol=0.0,
    )

    eigenvalues = torch.linalg.eigvalsh(
        result.risk_matrix
    )

    assert eigenvalues.min().item() >= -1e-10

    expected_diagonal = torch.diagonal(
        result.residual_covariance,
        dim1=-2,
        dim2=-1,
    )

    actual_diagonal = torch.diagonal(
        result.risk_matrix,
        dim1=-2,
        dim2=-1,
    )

    assert torch.allclose(
        actual_diagonal,
        expected_diagonal,
        atol=1e-12,
        rtol=1e-10,
    )


def test_joint_downside_tail_pair_has_stronger_dependence():
    torch.manual_seed(11)

    base = torch.randn(
        60,
        dtype=torch.float64,
    )

    asset_a = base
    asset_b = (
        base
        + 0.02
        * torch.randn_like(base)
    )

    permutation = torch.randperm(60)
    asset_c = base[permutation]

    residuals = torch.stack(
        [
            asset_a,
            asset_b,
            asset_c,
        ],
        dim=-1,
    ).unsqueeze(0)

    result = build_rank_tail_residual_risk(
        residuals=residuals,
        tail_q=0.20,
    )

    dependence = (
        result.rank_tail_dependence[0]
    )

    assert (
        dependence[0, 1]
        > dependence[0, 2]
    )


def test_rank_dependence_is_invariant_to_positive_asset_scaling():
    torch.manual_seed(17)

    residuals = torch.randn(
        2,
        60,
        4,
        dtype=torch.float64,
    )

    scales = torch.tensor(
        [0.5, 2.0, 5.0, 10.0],
        dtype=torch.float64,
    )

    original = build_rank_tail_residual_risk(
        residuals=residuals,
        tail_q=0.20,
    )

    scaled = build_rank_tail_residual_risk(
        residuals=(
            residuals
            * scales.view(
                1,
                1,
                -1,
            )
        ),
        tail_q=0.20,
    )

    assert torch.allclose(
        original.rank_tail_dependence,
        scaled.rank_tail_dependence,
        atol=1e-12,
        rtol=0.0,
    )


def test_tail_q_validation():
    residuals = torch.randn(
        1,
        60,
        3,
        dtype=torch.float64,
    )

    for q in [0.0, 1.0, -0.1, 1.1]:
        try:
            build_rank_tail_residual_risk(
                residuals=residuals,
                tail_q=q,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"tail_q={q} should fail"
            )
