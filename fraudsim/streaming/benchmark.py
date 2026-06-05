from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from confluent_kafka import Consumer, Producer, TopicPartition

from fraudsim.config import dataset_dir, load_config
from fraudsim.streaming.producer import emit_flush_markers
from fraudsim.streaming.topics import ensure_topics


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    bootstrap = args.bootstrap_servers or config["kafka"]["bootstrap_servers"]
    tx_topic = args.transaction_topic or config["topics"]["transaction_events"]
    risk_topic = args.risk_topic or config["topics"]["risk_results"]
    ensure_topics(
        bootstrap,
        [tx_topic, risk_topic, config["topics"]["alert_events"], config["topics"].get("late_events", "late_events")],
        args.partitions,
    )

    ds_dir = dataset_dir(config, args.dataset)
    stream_path = Path(args.stream_path) if args.stream_path else ds_dir / "transaction_stream.jsonl"
    producer = Producer({"bootstrap.servers": bootstrap})
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": f"fraudsim-benchmark-{int(time.time())}",
        "auto.offset.reset": "latest",
    })
    metadata = consumer.list_topics(topic=risk_topic, timeout=10)
    assignments: list[TopicPartition] = []
    for partition in sorted(metadata.topics[risk_topic].partitions):
        tp = TopicPartition(risk_topic, partition)
        _, high = consumer.get_watermark_offsets(tp, timeout=10)
        assignments.append(TopicPartition(risk_topic, partition, high))
    consumer.assign(assignments)

    sent_ids: set[str] = set()
    start = time.time()
    with stream_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            event = json.loads(line)
            event["_benchmark_sent_at"] = time.time()
            tx_id = str(event.get("transaction_id") or i)
            sent_ids.add(tx_id)
            producer.produce(
                tx_topic,
                key=str(event.get("payer_id") or tx_id).encode("utf-8"),
                value=json.dumps(event, ensure_ascii=False).encode("utf-8"),
            )
            producer.poll(0)
            if args.rate > 0:
                time.sleep(1 / args.rate)
    if args.emit_flush_markers:
        emit_flush_markers(producer, tx_topic, args.partitions)
    producer.flush()
    produce_done = time.time()

    received = 0
    latencies: list[float] = []
    deadline = time.time() + args.timeout
    while time.time() < deadline and received < len(sent_ids):
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        try:
            result = json.loads(msg.value().decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            continue
        tx_id = str(result.get("transaction_id"))
        if tx_id not in sent_ids:
            continue
        received += 1
        sent_at = result.get("_benchmark_sent_at")
        if sent_at:
            latencies.append(time.time() - float(sent_at))
    consumer.close()

    elapsed = max(time.time() - start, 1e-6)
    produce_elapsed = max(produce_done - start, 1e-6)
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1] if latencies_sorted else None
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99) - 1] if latencies_sorted else None
    report = {
        "sent": len(sent_ids),
        "received": received,
        "produce_events_per_sec": len(sent_ids) / produce_elapsed,
        "end_to_end_events_per_sec": received / elapsed,
        "latency_avg_sec": mean(latencies) if latencies else None,
        "latency_p95_sec": p95,
        "latency_p99_sec": p99,
        "timeout_sec": args.timeout,
        "flush_markers": args.emit_flush_markers,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small end-to-end Kafka/Flink/API streaming benchmark.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bootstrap-servers", default=None)
    parser.add_argument("--transaction-topic", default=None)
    parser.add_argument("--risk-topic", default=None)
    parser.add_argument("--stream-path", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--rate", type=float, default=0.0, help="Optional input rate in events/sec. Default sends a burst.")
    parser.add_argument("--emit-flush-markers", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
