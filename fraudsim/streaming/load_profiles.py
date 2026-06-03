from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
import redis

from fraudsim.config import dataset_dir, load_config
from fraudsim.graph_features import build_graph_entity_features


PROFILE_SPECS = [
    ("user_profile.parquet", "user_id", "profile:user"),
    ("merchant_profile.parquet", "merchant_id", "profile:merchant"),
    ("device_profile.parquet", "device_id", "profile:device"),
    ("ip_geo_profile.parquet", "ip_id", "profile:ip"),
]


def normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ds_dir = dataset_dir(config, args.dataset)
    redis_url = args.redis_url or config["redis"]["url"]
    client = redis.from_url(redis_url, decode_responses=True)

    for filename, key_col, prefix in PROFILE_SPECS:
        path = ds_dir / filename
        df = pd.read_parquet(path)
        pipe = client.pipeline(transaction=False)
        written = 0
        for row in df.to_dict(orient="records"):
            entity_id = row.get(key_col)
            if entity_id is None:
                continue
            payload = {k: normalize_value(v) for k, v in row.items()}
            pipe.set(f"{prefix}:{entity_id}", json.dumps(payload, ensure_ascii=False))
            written += 1
            if written % args.batch_size == 0:
                pipe.execute()
        pipe.execute()
        print(f"[profiles] {filename} written={written}")

    if args.with_graph_features:
        graph_df = build_graph_entity_features(ds_dir, force=args.rebuild_graph_features)
        pipe = client.pipeline(transaction=False)
        written = 0
        for row in graph_df.to_dict(orient="records"):
            entity_id = row.get("entity_id")
            if entity_id is None:
                continue
            payload = {k: normalize_value(v) for k, v in row.items()}
            pipe.set(f"profile:graph:{entity_id}", json.dumps(payload, ensure_ascii=False))
            written += 1
            if written % args.batch_size == 0:
                pipe.execute()
        pipe.execute()
        print(f"[profiles] graph_entity_features written={written}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load FP-FraudSim static profiles into Redis.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--with-graph-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-graph-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
