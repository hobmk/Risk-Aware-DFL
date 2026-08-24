from __future__ import annotations

import argparse
import sys

import torch

from implementations.rcr_dfl.src.dataset import (
    RCRRollingMVODataset as BaseRCRRollingMVODataset,
)
import implementations.rcr_dfl.scripts.train_rcr_combined as base


PLACEBO_SEED: int | None = None
PARSED_ARGS: argparse.Namespace | None = None


class FixedPermutationPlaceboDataset(BaseRCRRollingMVODataset):
    """A_res의 ticker mapping만 fixed permutation으로 깨뜨리는 placebo dataset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if PLACEBO_SEED is None:
            raise RuntimeError("PLACEBO_SEED가 설정되지 않았습니다.")

        generator = torch.Generator(device="cpu").manual_seed(PLACEBO_SEED)
        permutation = torch.randperm(
            self.n_assets,
            generator=generator,
        )

        identity = torch.arange(self.n_assets)

        if torch.equal(permutation, identity):
            raise RuntimeError(
                "Placebo permutation이 identity입니다. "
                "다른 placebo seed를 사용하세요."
            )

        original = self.a_res_matrices

        permuted = (
            original.index_select(-2, permutation)
            .index_select(-1, permutation)
            .contiguous()
        )

        # --------------------------------------------------------------
        # Placebo sanity check
        # P A P^T 이므로 trace / eigenvalue / symmetry는 유지되어야 함
        # --------------------------------------------------------------
        original_first = original[0].to(torch.float64)
        placebo_first = permuted[0].to(torch.float64)

        trace_diff = (
            torch.trace(original_first)
            - torch.trace(placebo_first)
        ).abs().item()

        eig_original = torch.linalg.eigvalsh(original_first)
        eig_placebo = torch.linalg.eigvalsh(placebo_first)

        eigen_diff = (
            eig_original - eig_placebo
        ).abs().max().item()

        symmetry_error = (
            placebo_first
            - placebo_first.T
        ).abs().max().item()

        matrix_diff = (
            original_first
            - placebo_first
        ).abs().max().item()

        if matrix_diff <= 0:
            raise RuntimeError(
                "Placebo matrix가 original A_res와 동일합니다."
            )

        if trace_diff > 1e-8:
            raise RuntimeError(
                f"Placebo trace가 보존되지 않았습니다: {trace_diff:.3e}"
            )

        if eigen_diff > 1e-8:
            raise RuntimeError(
                f"Placebo eigenvalue가 보존되지 않았습니다: {eigen_diff:.3e}"
            )

        if symmetry_error > 1e-10:
            raise RuntimeError(
                f"Placebo matrix가 비대칭입니다: {symmetry_error:.3e}"
            )

        self.a_res_matrices = permuted
        self.placebo_permutation = permutation

        # base train script의 config.json에도 기록되도록 추가
        if PARSED_ARGS is not None:
            PARSED_ARGS.placebo_enabled = True
            PARSED_ARGS.placebo_type = "fixed_ticker_permutation"
            PARSED_ARGS.placebo_permutation_seed = PLACEBO_SEED
            PARSED_ARGS.placebo_permutation = permutation.tolist()
            PARSED_ARGS.placebo_a_res_definition = "P @ A_res @ P.T"

        print("\n" + "=" * 100)
        print("FIXED-PERMUTATION PLACEBO CHECK")
        print("=" * 100)
        print(f"placebo seed          : {PLACEBO_SEED}")
        print(f"assets                : {self.n_assets}")
        print(f"permutation           : {permutation.tolist()}")
        print(f"identity permutation  : {torch.equal(permutation, identity)}")
        print(f"max matrix difference : {matrix_diff:.3e}")
        print(f"trace difference      : {trace_diff:.3e}")
        print(f"max eigenvalue diff   : {eigen_diff:.3e}")
        print(f"symmetry error        : {symmetry_error:.3e}")
        print("PLACEBO CHECK          : PASS")
        print("=" * 100 + "\n")


def parse_placebo_args() -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    parser.add_argument(
        "--placebo-permutation-seed",
        type=int,
        required=True,
    )

    placebo_args, remaining = parser.parse_known_args()

    if placebo_args.placebo_permutation_seed < 0:
        raise ValueError(
            "placebo-permutation-seed는 0 이상의 정수여야 합니다."
        )

    return (
        placebo_args.placebo_permutation_seed,
        remaining,
    )


def main() -> None:
    global PLACEBO_SEED
    global PARSED_ARGS

    placebo_seed, remaining = parse_placebo_args()
    PLACEBO_SEED = placebo_seed

    # --------------------------------------------------------------
    # 기존 train_rcr_combined가 나머지 argument를 그대로 처리하도록 함
    # --------------------------------------------------------------
    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    original_parse_args = base.parse_args

    def patched_parse_args() -> argparse.Namespace:
        global PARSED_ARGS

        args = original_parse_args()

        args.placebo_enabled = True
        args.placebo_type = "fixed_ticker_permutation"
        args.placebo_permutation_seed = PLACEBO_SEED
        args.placebo_permutation = None
        args.placebo_a_res_definition = "P @ A_res @ P.T"

        PARSED_ARGS = args
        return args

    # train_rcr_combined의 학습/검증/report 코드는 그대로 재사용
    base.parse_args = patched_parse_args
    base.RCRRollingMVODataset = FixedPermutationPlaceboDataset

    base.main()


if __name__ == "__main__":
    main()