from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from fraudsim.config import dataset_dir, load_config


def ensure_topic(bootstrap_servers: str, topic: str, partitions: int = 3) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    existing = set(admin.list_topics(timeout=10).topics)
    if topic in existing:
        return
    futures = admin.create_topics([NewTopic(topic, num_partitions=partitions, replication_factor=1)])
    futures[topic].result(timeout=30)


def delivery_report(err: Any, msg: Any) -> None:
    if err is not None:
        print(f"[producer] delivery failed: {err}")


def emit_flush_markers(producer: Producer, topic: str, partitions: int) -> None:
    for partition in range(max(1, partitions)):
        marker = {
            "__fraudsim_flush": True,
            "partition": partition,
            "sent_at": time.time(),
        }
        producer.produce(
            topic=topic,
            partition=partition,
            key=f"__fraudsim_flush_{partition}".encode("utf-8"),
            value=json.dumps(marker, ensure_ascii=False).encode("utf-8"),
            callback=delivery_report,
        )
        producer.poll(0)


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ds_dir = dataset_dir(config, args.dataset)
    stream_path = ds_dir / "transaction_stream.jsonl"
    topic = args.topic or config["topics"]["transaction_events"]
    bootstrap = args.bootstrap_servers or config["kafka"]["bootstrap_servers"]

    ensure_topic(bootstrap, topic, partitions=args.partitions)
    producer = Producer({"bootstrap.servers": bootstrap})
    delay = 0 if args.rate <= 0 else 1.0 / args.rate

    print(f"[producer] stream={stream_path}")
    print(f"[producer] topic={topic} bootstrap={bootstrap} rate={args.rate}/s")
    sent = 0
    with stream_path.open("r", encoding="utf-8") as f:
        for line in f:
            if args.limit and sent >= args.limit:
                break
            event = json.loads(line)
            key = str(event.get("payer_id") or event.get("transaction_id") or "")
            producer.produce(
                topic=topic,
                key=key.encode("utf-8"),
                value=json.dumps(event, ensure_ascii=False).encode("utf-8"),
                callback=delivery_report,
            )
            sent += 1
            producer.poll(0)
            if delay:
                time.sleep(delay)
            if sent % 1000 == 0:
                producer.flush(5)
                print(f"[producer] sent={sent}")
    if args.emit_flush_markers:
        emit_flush_markers(producer, topic, args.partitions)
    producer.flush()
    print(f"[producer] completed sent={sent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay FP-FraudSim transaction_stream.jsonl to Kafka.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bootstrap-servers", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--rate", type=float, default=100.0, help="Events per second; <=0 sends as fast as possible.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--emit-flush-markers", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
