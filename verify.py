"""Проверка CSV, независимых тестовых признаков и сохранённых моделей."""

from __future__ import annotations

import argparse
from pathlib import Path

from bot_solution.pipeline import REPOSITORY_ROOT, verify_saved_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs")
    parser.add_argument("--submission", type=Path, default=REPOSITORY_ROOT / "submission.csv")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    result = verify_saved_result(args.data_dir, args.output_dir, args.submission, args.threads)
    print(f"Verified {result['n_cookies']} scores from {result['n_models']} native models to 1e-12")
    print(f"Independent test-only rebuild matches all {result['n_features']} features")
    print(f"Exact submitted-v2 SHA-256: {result['submission_sha256']}")


if __name__ == "__main__":
    main()
