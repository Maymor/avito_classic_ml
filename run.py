"""Получить submission.csv с нуля: python run.py --data-dir /path/to/data."""

from __future__ import annotations

import argparse
from pathlib import Path

from bot_solution.pipeline import REPOSITORY_ROOT, reproduce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs")
    parser.add_argument("--submission", type=Path, default=REPOSITORY_ROOT / "submission.csv")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate", action="store_true", help="Also reproduce the three temporal CV blocks")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    reproduce(args.data_dir, args.output_dir, args.submission, args.threads, args.validate)


if __name__ == "__main__":
    main()
