from __future__ import annotations

import argparse
from pathlib import Path

from fraudsim.training.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain FP-FraudSim with reviewed feedback labels.")
    parser.add_argument("--feedback-path", default="data/feedback/feedback_pool.parquet")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--high-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feedback_path = Path(args.feedback_path)
    if not feedback_path.exists():
        raise SystemExit(f"feedback file not found: {feedback_path}")

    train(argparse.Namespace(
        config=args.config,
        dataset=args.dataset,
        model=args.model,
        output=args.output,
        high_threshold=args.high_threshold,
        with_window_features=True,
        with_graph_features=True,
        rebuild_graph_features=False,
        feedback_path=str(feedback_path),
    ))


if __name__ == "__main__":
    main()
