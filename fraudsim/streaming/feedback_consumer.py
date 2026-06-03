from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Consumer

from fraudsim.config import load_config


def normalize_feedback(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if "reviewed_is_fraud" not in out:
        if "is_fraud" in out:
            out["reviewed_is_fraud"] = out["is_fraud"]
        elif "label" in out:
            out["reviewed_is_fraud"] = out["label"]
    out["feedback_ingested_at"] = datetime.now(timezone.utc).isoformat()
    return out


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    bootstrap = args.bootstrap_servers or config["kafka"]["bootstrap_servers"]
    topic = args.topic or config["topics"].get("feedback_events", "feedback_events")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": args.group_id,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([topic])
    rows: list[dict[str, Any]] = []
    print(f"[feedback] consuming topic={topic} bootstrap={bootstrap}")
    try:
        while args.limit <= 0 or len(rows) < args.limit:
            msg = consumer.poll(args.poll_timeout)
            if msg is None:
                if args.exit_on_idle:
                    break
                continue
            if msg.error():
                print(f"[feedback] consumer error: {msg.error()}")
                continue
            rows.append(normalize_feedback(json.loads(msg.value().decode("utf-8"))))
            if len(rows) % args.flush_size == 0:
                flush(out_path, rows)
                rows.clear()
    finally:
        consumer.close()
    if rows:
        flush(out_path, rows)
    print(f"[feedback] written={out_path}")


def flush(path: Path, rows: list[dict[str, Any]]) -> None:
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_parquet(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist reviewed feedback events from Kafka to parquet.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--bootstrap-servers", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--group-id", default="fraudsim-feedback-writer")
    parser.add_argument("--output", default="data/feedback/feedback_pool.parquet")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--poll-timeout", type=float, default=2.0)
    parser.add_argument("--flush-size", type=int, default=1000)
    parser.add_argument("--exit-on-idle", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
