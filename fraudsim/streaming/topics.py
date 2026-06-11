from __future__ import annotations

import argparse

from confluent_kafka.admin import AdminClient, NewTopic

from fraudsim.config import load_config


def ensure_topics(bootstrap: str, topics: list[str], partitions: int) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    missing = [topic for topic in topics if topic not in existing]
    if not missing:
        print("[topics] all topics already exist")
        return
    futures = admin.create_topics([
        NewTopic(topic, num_partitions=partitions, replication_factor=1)
        for topic in missing
    ])
    for topic, future in futures.items():
        future.result(timeout=30)
        print(f"[topics] created {topic}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FP-FraudSim Kafka topics.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--bootstrap-servers", default=None)
    parser.add_argument("--partitions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bootstrap = args.bootstrap_servers or config["kafka"]["bootstrap_servers"]
    topics = [
        config["topics"]["transaction_events"],
        config["topics"]["risk_results"],
        config["topics"].get("risk_results_batch", "risk_results_batch"),
        config["topics"]["alert_events"],
        config["topics"].get("feedback_events", "feedback_events"),
        config["topics"].get("audit_events", "audit_events"),
        config["topics"].get("late_events", "late_events"),
    ]
    ensure_topics(bootstrap, topics, args.partitions)


if __name__ == "__main__":
    main()
