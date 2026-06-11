from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import requests
from confluent_kafka import Consumer, Producer

from fraudsim.config import load_config
from fraudsim.streaming.flink_job import WindowScorer
from fraudsim.streaming.topics import ensure_topics


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    bootstrap = args.bootstrap_servers or config["kafka"]["bootstrap_servers"]
    input_topic = args.input_topic or config["topics"]["transaction_events"]
    risk_topic = args.risk_topic or config["topics"].get("risk_results_batch", "risk_results_batch")
    alert_topic = args.alert_topic or config["topics"]["alert_events"]
    api_url = args.api_url or config["api"]["url"]
    redis_url = args.redis_url or config["redis"]["url"]

    ensure_topics(bootstrap, [input_topic, risk_topic, alert_topic], args.partitions)
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": args.group_id,
        "auto.offset.reset": args.offset_reset,
        "enable.auto.commit": False,
    })
    producer = Producer({"bootstrap.servers": bootstrap})
    scorer = WindowScorer(
        api_url=api_url,
        redis_url=redis_url,
        watermark_seconds=args.watermark_seconds,
        api_key=args.api_key,
    )
    session = requests.Session()
    if args.api_key:
        session.headers.update({"X-API-Key": args.api_key})
    consumer.subscribe([input_topic])
    print(f"[batch-risk] input={input_topic} risk={risk_topic} bootstrap={bootstrap} batch_size={args.batch_size}")

    processed = 0
    try:
        while args.limit <= 0 or processed < args.limit:
            batch: list[tuple[Any, dict[str, Any]]] = []
            deadline = time.time() + args.linger_ms / 1000
            while len(batch) < args.batch_size and time.time() < deadline:
                msg = consumer.poll(0.05)
                if msg is None:
                    continue
                if msg.error():
                    print(f"[batch-risk] consumer error: {msg.error()}")
                    continue
                event = json.loads(msg.value().decode("utf-8"))
                batch.append((msg, scorer.enrich(event)))
            if not batch:
                if args.exit_on_idle:
                    break
                continue

            records = [record for _, record in batch]
            response = session.post(f"{api_url.rstrip('/')}/predict", json={"records": records}, timeout=args.api_timeout)
            response.raise_for_status()
            results = response.json()["results"]

            for (msg, enriched), scored in zip(batch, results):
                output = dict(enriched)
                output.update(scored)
                raw = json.dumps(output, ensure_ascii=False).encode("utf-8")
                key = str(output.get("payer_id") or output.get("transaction_id") or "").encode("utf-8")
                producer.produce(risk_topic, key=key, value=raw)
                if output.get("risk_level") == "high" or output.get("decision") == "reject":
                    producer.produce(alert_topic, key=key, value=raw)
            producer.flush(5)
            consumer.commit(asynchronous=False)
            processed += len(batch)
            if processed % args.log_every == 0 or (args.limit > 0 and processed >= args.limit):
                print(f"[batch-risk] processed={processed}")
    finally:
        consumer.close()
        producer.flush()
    print(f"[batch-risk] completed processed={processed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-batch Kafka risk worker using FastAPI batch /predict.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--bootstrap-servers", default=None)
    parser.add_argument("--input-topic", default=None)
    parser.add_argument("--risk-topic", default=None)
    parser.add_argument("--alert-topic", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--group-id", default="fraudsim-batch-risk")
    parser.add_argument("--offset-reset", default="latest")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--linger-ms", type=int, default=500)
    parser.add_argument("--api-timeout", type=float, default=30.0)
    parser.add_argument("--api-key", default=os.getenv("FRAUDSIM_API_KEY") or None)
    parser.add_argument("--watermark-seconds", type=int, default=60)
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--exit-on-idle", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
