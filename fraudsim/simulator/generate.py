from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fraudsim.config import dataset_dir, load_config


@dataclass
class SimulationConfig:
    dataset: str
    seed: int
    rows: int
    fraud_ratio: float
    fraud_group_count: int
    fraud_group_size: int
    shared_device_strength: float
    shared_ip_strength: float
    cross_border_ratio: float
    large_amount_ratio: float
    night_ratio: float
    rate: int
    generated_at: str


def read_base_stream(ds_dir: Path, rows: int) -> list[dict[str, Any]]:
    path = ds_dir / "transaction_stream.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"transaction stream not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= rows:
                break
    if not out:
        raise ValueError(f"no records found in {path}")
    return out


def choose_group(index: int, config: SimulationConfig) -> dict[str, str]:
    group_no = index % max(1, config.fraud_group_count)
    member_no = index % max(1, config.fraud_group_size)
    return {
        "fraud_group_id": f"SIM_G{group_no:04d}",
        "payer_id": f"SIM_U_G{group_no:04d}_{member_no:03d}",
        "device_id": f"SIM_D_G{group_no:04d}_{member_no % max(1, int(config.fraud_group_size * (1 - config.shared_device_strength)) + 1):03d}",
        "ip_id": f"SIM_IP_G{group_no:04d}_{member_no % max(1, int(config.fraud_group_size * (1 - config.shared_ip_strength)) + 1):03d}",
        "payee_id": f"SIM_PAYEE_G{group_no:04d}_{member_no % 3:03d}",
        "merchant_id": f"SIM_M_G{group_no:04d}_{member_no % 2:03d}",
    }


def mutate_fraud_record(record: dict[str, Any], fraud_index: int, config: SimulationConfig, rng: random.Random) -> dict[str, Any]:
    out = dict(record)
    group = choose_group(fraud_index, config)
    out.update(group)
    out["is_fraud"] = 1
    out["label_quality"] = "simulated_fraud"
    out["injected_scenario"] = rng.choice([
        "shared_device_ring",
        "shared_ip_cluster",
        "cross_border_cashout",
        "merchant_cashout_cluster",
        "mule_transfer_chain",
    ])
    if rng.random() < config.large_amount_ratio:
        out["amount"] = max(float(out.get("amount") or 0), rng.uniform(5000, 30000))
        out["amount_bucket"] = "large"
    if rng.random() < config.cross_border_ratio:
        out["payer_country"] = rng.choice(["CN", "SG", "MY", "TH"])
        out["payee_country"] = rng.choice(["US", "GB", "AE", "HK"])
    if rng.random() < config.night_ratio:
        out["hour"] = rng.choice([0, 1, 2, 3, 4, 23])
        out["is_night"] = 1
    out["simulation_tag"] = "generated_fraud"
    return out


def normalize_record(record: dict[str, Any], index: int, start_time: datetime, config: SimulationConfig) -> dict[str, Any]:
    out = dict(record)
    out["transaction_id"] = f"SIM_TXN_{index:08d}"
    event_time = start_time + timedelta(seconds=index / max(1, config.rate))
    out["timestamp"] = event_time.replace(tzinfo=None).isoformat(timespec="seconds")
    out.setdefault("source", "simulator")
    out.setdefault("is_fraud", 0)
    out.setdefault("simulation_tag", "generated_normal")
    return out


def generate(config: SimulationConfig, ds_dir: Path, output_dir: Path, replace_stream: bool = False) -> dict[str, Any]:
    rng = random.Random(config.seed)
    base = read_base_stream(ds_dir, config.rows)
    start_time = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    fraud_count = int(config.rows * config.fraud_ratio)
    fraud_indexes = set(rng.sample(range(config.rows), min(fraud_count, config.rows)))
    rows: list[dict[str, Any]] = []
    fraud_seen = 0
    for index in range(config.rows):
        record = normalize_record(base[index % len(base)], index, start_time, config)
        if index in fraud_indexes:
            record = mutate_fraud_record(record, fraud_seen, config, rng)
            fraud_seen += 1
        else:
            record["is_fraud"] = 0
        rows.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = output_dir / "transaction_stream.jsonl"
    with stream_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    labels = pd.DataFrame([
        {
            "transaction_id": row["transaction_id"],
            "is_fraud": int(row.get("is_fraud", 0)),
            "fraud_group_id": row.get("fraud_group_id"),
            "injected_scenario": row.get("injected_scenario"),
        }
        for row in rows
    ])
    labels.to_parquet(output_dir / "simulation_labels.parquet", index=False)
    with (output_dir / "simulation_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    if replace_stream:
        shutil.copy2(stream_path, ds_dir / "transaction_stream.jsonl")

    return {
        "stream_path": str(stream_path),
        "labels_path": str(output_dir / "simulation_labels.parquet"),
        "config_path": str(output_dir / "simulation_config.json"),
        "rows": len(rows),
        "fraud_rows": int(labels["is_fraud"].sum()),
        "fraud_ratio": float(labels["is_fraud"].mean()),
        "replace_stream": replace_stream,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate configurable simulated FP-FraudSim transaction streams.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--fraud-ratio", type=float, default=0.03)
    parser.add_argument("--fraud-group-count", type=int, default=20)
    parser.add_argument("--fraud-group-size", type=int, default=12)
    parser.add_argument("--shared-device-strength", type=float, default=0.75)
    parser.add_argument("--shared-ip-strength", type=float, default=0.70)
    parser.add_argument("--cross-border-ratio", type=float, default=0.65)
    parser.add_argument("--large-amount-ratio", type=float, default=0.55)
    parser.add_argument("--night-ratio", type=float, default=0.35)
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--replace-stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ds_dir = dataset_dir(cfg, args.dataset)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ds_dir / "simulations" / stamp
    config = SimulationConfig(
        dataset=ds_dir.name,
        seed=args.seed,
        rows=args.rows,
        fraud_ratio=args.fraud_ratio,
        fraud_group_count=args.fraud_group_count,
        fraud_group_size=args.fraud_group_size,
        shared_device_strength=args.shared_device_strength,
        shared_ip_strength=args.shared_ip_strength,
        cross_border_ratio=args.cross_border_ratio,
        large_amount_ratio=args.large_amount_ratio,
        night_ratio=args.night_ratio,
        rate=args.rate,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    summary = generate(config, ds_dir, output_dir, replace_stream=args.replace_stream)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

