from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import requests

from fraudsim.window_features import compute_window_feature_dict, parse_timestamp


GRAPH_MINING_FIELDS = (
    "fraud_group_mining_id",
    "graph_mining_group_risk_score",
    "graph_mining_group_entity_count",
    "graph_mining_group_user_count",
    "graph_mining_group_resource_count",
    "graph_mining_fraud_seed_count",
    "graph_mining_fraud_seed_ratio",
    "graph_mining_shared_device_count",
    "graph_mining_shared_ip_count",
    "graph_mining_shared_merchant_count",
    "graph_mining_shared_payee_count",
    "graph_mining_evidence_count",
    "graph_mining_scenario_count",
)


def parse_ts(value: str | None) -> float:
    return parse_timestamp(value)


def is_flush_marker(value: str) -> bool:
    try:
        event = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(event, dict) and bool(event.get("__fraudsim_flush"))


class WindowScorer:
    def __init__(self, api_url: str, redis_url: str, watermark_seconds: int = 60) -> None:
        self.api_url = api_url.rstrip("/")
        self.redis_url = redis_url
        self.watermark_seconds = watermark_seconds
        self.max_event_ts = 0.0
        self.redis_client = None
        self.http_session = requests.Session()
        self.user_events: dict[str, deque] = defaultdict(deque)
        self.device_events: dict[str, deque] = defaultdict(deque)
        self.ip_events: dict[str, deque] = defaultdict(deque)
        self.merchant_events: dict[str, deque] = defaultdict(deque)

    @staticmethod
    def _trim(events: deque, now_ts: float, seconds: int) -> None:
        while events and now_ts - events[0]["ts"] > seconds:
            events.popleft()

    @staticmethod
    def _sum_amount(events: deque) -> float:
        return float(sum(float(e.get("amount") or 0.0) for e in events))

    @staticmethod
    def _unique(events: deque, key: str) -> int:
        return len({e.get(key) for e in events if e.get(key)})

    def _profile(self, prefix: str, entity_id: str | None) -> dict[str, Any]:
        if self.redis_client is None:
            import redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        if not entity_id:
            return {}
        raw = self.redis_client.get(f"profile:{prefix}:{entity_id}")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _graph_profile(self, entity_id: str | None, prefix: str, fields: tuple[str, ...]) -> dict[str, Any]:
        raw = self._profile("graph", entity_id)
        return {f"{prefix}_{field}": raw.get(field, 0.0) for field in fields}

    def enrich(self, event: dict[str, Any]) -> dict[str, Any]:
        event_ts = parse_ts(event.get("timestamp"))
        self.max_event_ts = max(self.max_event_ts, event_ts)
        is_late_event = event_ts < self.max_event_ts - self.watermark_seconds

        payer = str(event.get("payer_id") or "")
        device = str(event.get("device_id") or "")
        ip = str(event.get("ip_id") or "")
        merchant = str(event.get("merchant_id") or "")
        payee = str(event.get("payee_id") or "")
        window_features = compute_window_feature_dict(
            event,
            self.user_events,
            self.device_events,
            self.ip_events,
            self.merchant_events,
        )
        enriched = dict(event)
        enriched["event_time_unix"] = event_ts
        enriched["is_late_event"] = is_late_event
        enriched["window_features"] = window_features
        enriched["user_profile"] = self._profile("user", payer)
        enriched["device_profile"] = self._profile("device", device)
        enriched["ip_profile"] = self._profile("ip", ip)
        enriched["merchant_profile"] = self._profile("merchant", merchant)
        enriched["graph_features"] = {
            **self._graph_profile(payer, "payer", ("graph_degree", "graph_fraud_edge_count", "graph_fraud_edge_ratio", *GRAPH_MINING_FIELDS)),
            **self._graph_profile(payee, "payee", ("graph_degree", "graph_mining_group_risk_score")),
            **self._graph_profile(merchant, "merchant", ("graph_degree", "graph_mining_group_risk_score")),
            **self._graph_profile(device, "device", ("graph_degree", "graph_mining_group_risk_score")),
            **self._graph_profile(ip, "ip", ("graph_degree", "graph_mining_group_risk_score")),
        }
        return enriched

    def __call__(self, value: str) -> str:
        event = json.loads(value)
        enriched = self.enrich(event)
        response = self.http_session.post(f"{self.api_url}/predict", json={"record": enriched}, timeout=5)
        response.raise_for_status()
        scored = response.json()["results"][0]
        output = dict(enriched)
        output.update(scored)
        return json.dumps(output, ensure_ascii=False)


class MicroBatchWindowScorer(WindowScorer):
    def __init__(
        self,
        api_url: str,
        redis_url: str,
        watermark_seconds: int = 60,
        batch_size: int = 50,
        linger_ms: int = 300,
    ) -> None:
        super().__init__(api_url=api_url, redis_url=redis_url, watermark_seconds=watermark_seconds)
        self.batch_size = max(1, batch_size)
        self.linger_seconds = max(0, linger_ms) / 1000
        self.buffer: list[dict[str, Any]] = []
        self.last_flush = time.monotonic()

    def __call__(self, value: str) -> list[str]:
        return self.flat_map(value)

    def _flush(self) -> list[str]:
        if not self.buffer:
            return []
        records = self.buffer
        self.buffer = []
        self.last_flush = time.monotonic()

        response = self.http_session.post(f"{self.api_url}/predict", json={"records": records}, timeout=15)
        response.raise_for_status()
        scored_rows = response.json()["results"]
        outputs: list[str] = []
        for enriched, scored in zip(records, scored_rows):
            output = dict(enriched)
            output.update(scored)
            outputs.append(json.dumps(output, ensure_ascii=False))
        return outputs

    def flat_map(self, value: str) -> list[str]:
        event = json.loads(value)
        if event.get("__fraudsim_flush"):
            return self._flush()
        self.buffer.append(self.enrich(event))
        should_flush_by_size = len(self.buffer) >= self.batch_size
        should_flush_by_time = self.linger_seconds > 0 and time.monotonic() - self.last_flush >= self.linger_seconds
        if should_flush_by_size or should_flush_by_time:
            return self._flush()
        return []


def main() -> None:
    from pyflink.common import SimpleStringSchema, Types
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer

    bootstrap = os.getenv("FRAUDSIM_KAFKA_BOOTSTRAP", "kafka:9092")
    input_topic = os.getenv("FRAUDSIM_TOPIC_IN", "transaction_events")
    risk_topic = os.getenv("FRAUDSIM_TOPIC_RISK", "risk_results")
    alert_topic = os.getenv("FRAUDSIM_TOPIC_ALERT", "alert_events")
    late_topic = os.getenv("FRAUDSIM_TOPIC_LATE", "late_events")
    group_id = os.getenv("FRAUDSIM_FLINK_GROUP", "fraudsim-flink")
    api_url = os.getenv("FRAUDSIM_API_URL", "http://model-api:8000")
    redis_url = os.getenv("FRAUDSIM_REDIS_URL", "redis://redis:6379/0")
    watermark_seconds = int(os.getenv("FRAUDSIM_WATERMARK_SECONDS", "60"))
    scoring_mode = os.getenv("FRAUDSIM_SCORING_MODE", "sync").lower()
    batch_size = int(os.getenv("FRAUDSIM_BATCH_SIZE", "50"))
    linger_ms = int(os.getenv("FRAUDSIM_BATCH_LINGER_MS", "300"))

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(int(os.getenv("FRAUDSIM_FLINK_PARALLELISM", "1")))
    checkpoint_ms = int(os.getenv("FRAUDSIM_FLINK_CHECKPOINT_MS", "60000"))
    if checkpoint_ms > 0:
        env.enable_checkpointing(checkpoint_ms)

    consumer = FlinkKafkaConsumer(
        topics=input_topic,
        deserialization_schema=SimpleStringSchema(),
        properties={
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        },
    )
    stream = env.add_source(consumer)
    if scoring_mode == "microbatch":
        scored = stream.flat_map(
            MicroBatchWindowScorer(
                api_url=api_url,
                redis_url=redis_url,
                watermark_seconds=watermark_seconds,
                batch_size=batch_size,
                linger_ms=linger_ms,
            ),
            output_type=Types.STRING(),
        )
    else:
        scored = stream.filter(lambda raw: not is_flush_marker(raw)).map(
            WindowScorer(api_url=api_url, redis_url=redis_url, watermark_seconds=watermark_seconds),
            output_type=Types.STRING(),
        )

    risk_producer = FlinkKafkaProducer(
        topic=risk_topic,
        serialization_schema=SimpleStringSchema(),
        producer_config={"bootstrap.servers": bootstrap},
    )
    alert_producer = FlinkKafkaProducer(
        topic=alert_topic,
        serialization_schema=SimpleStringSchema(),
        producer_config={"bootstrap.servers": bootstrap},
    )
    late_producer = FlinkKafkaProducer(
        topic=late_topic,
        serialization_schema=SimpleStringSchema(),
        producer_config={"bootstrap.servers": bootstrap},
    )

    scored.add_sink(risk_producer)
    scored.filter(lambda raw: json.loads(raw).get("risk_level") == "high" or json.loads(raw).get("decision") == "reject").add_sink(alert_producer)
    scored.filter(lambda raw: bool(json.loads(raw).get("is_late_event"))).add_sink(late_producer)
    env.execute(f"fp-fraudsim-streaming-risk-{scoring_mode}")


if __name__ == "__main__":
    main()
