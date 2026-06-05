from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def sample_record(index: int) -> dict[str, Any]:
    amount = 120 + (index % 17) * 530
    return {
        "transaction_id": f"latency_demo_{index}",
        "timestamp": "2026-03-20T10:15:03",
        "source": "demo",
        "amount": amount,
        "amount_bucket": "large" if amount >= 5000 else "medium",
        "currency": "USD",
        "txn_type": "transfer" if index % 3 == 0 else "payment",
        "channel": "app" if index % 2 == 0 else "mini_program",
        "payment_method": "bank_transfer",
        "merchant_category": "digital_goods",
        "payer_id": f"demo_user_{index % 25:03d}",
        "payee_id": f"demo_payee_{index % 40:03d}",
        "merchant_id": f"demo_merchant_{index % 12:03d}",
        "device_id": f"demo_device_{index % 15:03d}",
        "ip_id": f"demo_ip_{index % 20:03d}",
        "payer_country": "CN",
        "payee_country": "US",
        "hour": 10,
        "day_of_week": 4,
        "is_weekend": 0,
        "is_night": 0,
        "window_features": {
            "user_txn_count_5min": 1 + index % 8,
            "user_amount_sum_5min": amount * (1 + index % 4),
            "user_txn_count_1h": 2 + index % 15,
            "user_amount_sum_1h": amount * (2 + index % 8),
            "user_unique_payee_count_1h": 1 + index % 6,
            "device_unique_user_count_10min": 1 + index % 5,
            "ip_unique_user_count_10min": 1 + index % 7,
            "merchant_txn_count_1h": 3 + index % 60,
            "merchant_amount_sum_1h": amount * (3 + index % 10),
            "merchant_unique_user_count_1h": 2 + index % 30,
        },
        "user_profile": {
            "txn_count": 10 + index % 120,
            "total_amount": amount * 10,
            "avg_amount": max(1, amount / 2),
            "common_device_count": 1 + index % 8,
            "common_ip_count": 1 + index % 8,
            "home_country": "CN",
            "account_age_days": 60 + index % 900,
            "risk_history_score": (index % 100) / 100,
        },
        "device_profile": {
            "bind_user_count": 1 + index % 20,
            "txn_count": 1 + index % 80,
            "device_type": "mobile",
            "os": "Android",
            "is_emulator": 1 if index % 23 == 0 else 0,
            "is_proxy_device": 1 if index % 19 == 0 else 0,
        },
        "ip_profile": {
            "bind_user_count": 1 + index % 25,
            "country": "US",
            "is_proxy": 1 if index % 17 == 0 else 0,
            "is_vpn": 1 if index % 29 == 0 else 0,
        },
        "merchant_profile": {
            "merchant_category": "digital_goods",
            "merchant_country": "US",
            "txn_count": 20 + index % 500,
            "unique_user_count": 10 + index % 200,
            "total_amount": amount * 20,
            "avg_amount": max(1, amount / 3),
            "chargeback_rate": (index % 20) / 200,
            "complaint_rate": (index % 15) / 200,
            "merchant_risk_score": (index % 100) / 100,
        },
        "graph_features": {
            "payer_graph_degree": 1 + index % 100,
            "payer_graph_fraud_edge_count": index % 9,
            "payer_graph_fraud_edge_ratio": (index % 10) / 10,
            "payee_graph_degree": 1 + index % 120,
            "merchant_graph_degree": 1 + index % 300,
            "device_graph_degree": 1 + index % 50,
            "ip_graph_degree": 1 + index % 60,
        },
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((pct / 100) * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "avg_ms": statistics.mean(values) if values else 0.0,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    session = requests.Session()
    base_url = args.api_url.rstrip("/")
    health = session.get(f"{base_url}/health", timeout=10).json()
    records = [sample_record(i) for i in range(args.records)]

    single_latencies: list[float] = []
    for record in records:
        started = time.perf_counter()
        response = session.post(f"{base_url}/predict", json={"record": record}, timeout=10)
        response.raise_for_status()
        single_latencies.append((time.perf_counter() - started) * 1000)

    batch_latencies: list[float] = []
    batch_record_latencies: list[float] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        started = time.perf_counter()
        response = session.post(f"{base_url}/predict", json={"records": batch}, timeout=30)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - started) * 1000
        batch_latencies.append(elapsed_ms)
        batch_record_latencies.append(elapsed_ms / len(batch))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": base_url,
        "model": {
            "loaded": health.get("loaded"),
            "model_name": health.get("model_name"),
            "model_version": health.get("model_version"),
            "feature_count": health.get("feature_count"),
            "thresholds": health.get("thresholds"),
        },
        "records": len(records),
        "batch_size": args.batch_size,
        "single_record_api": summarize(single_latencies),
        "batch_api": {
            "batch_latency": summarize(batch_latencies),
            "per_record_latency": summarize(batch_record_latencies),
        },
        "speedup_avg_per_record": (
            statistics.mean(single_latencies) / statistics.mean(batch_record_latencies)
            if single_latencies and batch_record_latencies else 0.0
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-record and batch /predict API latency.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output", default="models/api_latency_microbatch_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
