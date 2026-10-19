from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank-Tail q grid"
    )

    parser.add_argument(
        "--q-values",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output-root",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
            "outputs/q_grid"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def is_completed(output_dir: Path) -> bool:
    required_files = [
        output_dir / "portfolio_summary.csv",
        output_dir / "test_metrics.csv",
        output_dir / "config.json",
        output_dir / "best_model.pt",
    ]

    return all(
        path.exists()
        for path in required_files
    )


def main():
    args = parse_args()

    runs = [
        (q, seed)
        for q in args.q_values
        for seed in args.seeds
    ]

    completed = 0
    skipped = 0

    for index, (q, seed) in enumerate(
        runs,
        start=1,
    ):
        output_dir = (
            Path(args.output_root)
            / f"q_{q:.2f}"
            / f"seed_{seed}"
        )

        print("\n" + "=" * 72)
        print(
            f"[{index}/{len(runs)}] "
            f"q={q:.2f} seed={seed}"
        )
        print("=" * 72)

        if (
            not args.overwrite
            and is_completed(output_dir)
        ):
            print(
                f"[SKIP] completed run: "
                f"{output_dir}"
            )

            skipped += 1
            continue

        if (
            output_dir.exists()
            and not args.overwrite
        ):
            print(
                f"[REBUILD] incomplete run found: "
                f"{output_dir}"
            )

            command_overwrite = True
        else:
            command_overwrite = (
                args.overwrite
            )

        command = [
            sys.executable,
            "-m",
            (
                "implementations."
                "rank_tail_rcr_dfl."
                "scripts."
                "train_rank_tail_rcr"
            ),
            "--tail-q",
            str(q),
            "--alpha",
            str(args.alpha),
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        ]

        if command_overwrite:
            command.append(
                "--overwrite"
            )

        subprocess.run(
            command,
            check=True,
        )

        completed += 1

    print("\n" + "=" * 72)
    print("GRID FINISHED")
    print(f"Executed : {completed}")
    print(f"Skipped  : {skipped}")
    print(f"Total    : {len(runs)}")
    print("=" * 72)


if __name__ == "__main__":
    main()