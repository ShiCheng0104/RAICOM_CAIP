from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraudsim.training.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain FP-FraudSim with reviewed feedback labels.")
    parser.add_argument("--feedback-path", default="data/feedback/feedback_pool.parquet")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--output", default=None, help="Defaults to models/{model}/candidate.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--high-threshold", type=float, default=0.80)
    parser.add_argument("--with-graph-mining-features", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feedback_path = Path(args.feedback_path)
    if not feedback_path.exists():
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.touch()
        print(f"[fraudsim] feedback file did not exist, created empty pool: {feedback_path}")

    output = args.output or str(Path("models") / args.model / "candidate")
    metrics = train(argparse.Namespace(
        config=args.config,
        dataset=args.dataset,
        model=args.model,
        output=output,
        high_threshold=args.high_threshold,
        with_window_features=True,
        with_graph_features=True,
        with_graph_mining_features=args.with_graph_mining_features,
        rebuild_graph_features=False,
        feedback_path=str(feedback_path),
    ))
    manifest = {
        "candidate_path": output,
        "feedback_path": str(feedback_path),
        "metrics": metrics,
    }
    with (Path(output) / "candidate_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
