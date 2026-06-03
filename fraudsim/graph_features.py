from __future__ import annotations

from pathlib import Path

import pandas as pd


GRAPH_FEATURE_COLUMNS = [
    "payer_graph_degree",
    "payer_graph_fraud_edge_count",
    "payer_graph_fraud_edge_ratio",
    "payee_graph_degree",
    "merchant_graph_degree",
    "device_graph_degree",
    "ip_graph_degree",
]


def graph_feature_path(dataset_dir: Path) -> Path:
    return dataset_dir / "graph_entity_features.parquet"


def build_graph_entity_features(dataset_dir: Path, force: bool = False) -> pd.DataFrame:
    out_path = graph_feature_path(dataset_dir)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    edges_path = dataset_dir / "graph_edges.parquet"
    if not edges_path.exists():
        return pd.DataFrame(columns=["entity_id", "graph_degree", "graph_fraud_edge_count", "graph_fraud_edge_ratio"])

    edges = pd.read_parquet(edges_path, columns=["src_id", "dst_id", "label"])
    src = edges[["src_id", "label"]].rename(columns={"src_id": "entity_id"})
    dst = edges[["dst_id", "label"]].rename(columns={"dst_id": "entity_id"})
    endpoints = pd.concat([src, dst], ignore_index=True)
    endpoints["is_fraud_edge"] = (pd.to_numeric(endpoints["label"], errors="coerce").fillna(0) == 1).astype("int8")

    features = endpoints.groupby("entity_id", as_index=False).agg(
        graph_degree=("entity_id", "size"),
        graph_fraud_edge_count=("is_fraud_edge", "sum"),
    )
    features["graph_fraud_edge_ratio"] = (
        features["graph_fraud_edge_count"] / features["graph_degree"].clip(lower=1)
    ).astype("float32")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    return features


def add_graph_features(df: pd.DataFrame, dataset_dir: Path, force_rebuild: bool = False) -> pd.DataFrame:
    graph_features = build_graph_entity_features(dataset_dir, force=force_rebuild)
    out = df.copy()
    if graph_features.empty:
        for col in GRAPH_FEATURE_COLUMNS:
            out[col] = 0.0
        return out

    for entity_col, prefix, cols in [
        ("payer_id", "payer", ["graph_degree", "graph_fraud_edge_count", "graph_fraud_edge_ratio"]),
        ("payee_id", "payee", ["graph_degree"]),
        ("merchant_id", "merchant", ["graph_degree"]),
        ("device_id", "device", ["graph_degree"]),
        ("ip_id", "ip", ["graph_degree"]),
    ]:
        if entity_col not in out.columns:
            continue
        selected = graph_features[["entity_id", *cols]].rename(
            columns={col: f"{prefix}_{col}" for col in cols}
        )
        out = out.merge(selected, left_on=entity_col, right_on="entity_id", how="left")
        out = out.drop(columns=["entity_id"])

    for col in GRAPH_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out
