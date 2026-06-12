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


def write_frame(
    client: redis.Redis,
    frame: pd.DataFrame,
    key_col: str,
    prefix: str,
    batch_size: int,
) -> int:
    columns = frame.columns.tolist()
    try:
        key_index = columns.index(key_col)
    except ValueError as exc:
        raise ValueError(f"Missing key column {key_col!r} for Redis prefix {prefix!r}") from exc

    pipe = client.pipeline(transaction=False)
    written = 0
    for values in frame.itertuples(index=False, name=None):
        entity_id = values[key_index]
        if pd.isna(entity_id):
            continue
        payload = {column: normalize_value(value) for column, value in zip(columns, values)}
        pipe.set(f"{prefix}:{entity_id}", json.dumps(payload, ensure_ascii=False))
        written += 1
        if written % batch_size == 0:
            pipe.execute()
    pipe.execute()
    return written


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ds_dir = dataset_dir(config, args.dataset)
    redis_url = args.redis_url or config["redis"]["url"]
    client = redis.from_url(redis_url, decode_responses=True)

    for filename, key_col, prefix in PROFILE_SPECS:
        path = ds_dir / filename
        df = pd.read_parquet(path)
        written = write_frame(client, df, key_col, prefix, args.batch_size)
        print(f"[profiles] {filename} written={written}")

    if args.with_graph_features:
        graph_df = build_graph_entity_features(ds_dir, force=args.rebuild_graph_features)
        written = write_frame(client, graph_df, "entity_id", "profile:graph", args.batch_size)
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
