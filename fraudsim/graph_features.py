from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudsim.graph_mining import ENTITY_FEATURE_COLUMNS, load_entity_graph_risk


BASE_GRAPH_FEATURE_COLUMNS = [
    "payer_graph_degree",
    "payer_graph_fraud_edge_count",
    "payer_graph_fraud_edge_ratio",
    "payee_graph_degree",
    "merchant_graph_degree",
    "device_graph_degree",
    "ip_graph_degree",
]

GRAPH_MINING_FEATURE_COLUMNS = [
    "payer_fraud_group_mining_id",
    "payer_graph_mining_group_risk_score",
    "payer_graph_mining_group_entity_count",
    "payer_graph_mining_group_user_count",
    "payer_graph_mining_group_resource_count",
    "payer_graph_mining_fraud_seed_count",
    "payer_graph_mining_fraud_seed_ratio",
    "payer_graph_mining_shared_device_count",
    "payer_graph_mining_shared_ip_count",
    "payer_graph_mining_shared_merchant_count",
    "payer_graph_mining_shared_payee_count",
    "payer_graph_mining_evidence_count",
    "payer_graph_mining_scenario_count",
    "payee_graph_mining_group_risk_score",
    "merchant_graph_mining_group_risk_score",
    "device_graph_mining_group_risk_score",
    "ip_graph_mining_group_risk_score",
]

GRAPH_FEATURE_COLUMNS = BASE_GRAPH_FEATURE_COLUMNS + GRAPH_MINING_FEATURE_COLUMNS

GRAPH_ENTITY_BASE_COLUMNS = [
    "entity_id",
    "graph_degree",
    "graph_fraud_edge_count",
    "graph_fraud_edge_ratio",
]

GRAPH_ENTITY_EXPECTED_COLUMNS = GRAPH_ENTITY_BASE_COLUMNS + ENTITY_FEATURE_COLUMNS


def graph_feature_path(dataset_dir: Path) -> Path:
    return dataset_dir / "graph_entity_features.parquet"


def build_graph_entity_features(dataset_dir: Path, force: bool = False) -> pd.DataFrame:
    out_path = graph_feature_path(dataset_dir)
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        if set(GRAPH_ENTITY_EXPECTED_COLUMNS).issubset(existing.columns):
            return existing

    edges_path = dataset_dir / "graph_edges.parquet"
    if not edges_path.exists():
        base_features = pd.DataFrame(columns=GRAPH_ENTITY_BASE_COLUMNS)
    else:
        edges = pd.read_parquet(edges_path, columns=["src_id", "dst_id", "label"])
        src = edges[["src_id", "label"]].rename(columns={"src_id": "entity_id"})
        dst = edges[["dst_id", "label"]].rename(columns={"dst_id": "entity_id"})
        endpoints = pd.concat([src, dst], ignore_index=True)
        endpoints["is_fraud_edge"] = (pd.to_numeric(endpoints["label"], errors="coerce").fillna(0) == 1).astype("int8")

        base_features = endpoints.groupby("entity_id", as_index=False).agg(
            graph_degree=("entity_id", "size"),
            graph_fraud_edge_count=("is_fraud_edge", "sum"),
        )
        base_features["graph_fraud_edge_ratio"] = (
            base_features["graph_fraud_edge_count"] / base_features["graph_degree"].clip(lower=1)
        ).astype("float32")

    mining_features = load_entity_graph_risk(dataset_dir, force=force)
    features = base_features.merge(mining_features, on="entity_id", how="outer")
    for col in GRAPH_ENTITY_BASE_COLUMNS:
        if col != "entity_id":
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    for col in ENTITY_FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = None if col == "fraud_group_mining_id" else 0.0
        elif col != "fraud_group_mining_id":
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    return features


def add_graph_features(
    df: pd.DataFrame,
    dataset_dir: Path,
    force_rebuild: bool = False,
    include_graph_mining: bool = False,
) -> pd.DataFrame:
    graph_features = build_graph_entity_features(dataset_dir, force=force_rebuild)
    out = df.copy()
    expected_columns = GRAPH_FEATURE_COLUMNS if include_graph_mining else BASE_GRAPH_FEATURE_COLUMNS
    if graph_features.empty:
        for col in expected_columns:
            out[col] = 0.0
        return out

    mining_cols = ["fraud_group_mining_id", *[col for col in ENTITY_FEATURE_COLUMNS if col != "fraud_group_mining_id"]]
    payer_cols = ["graph_degree", "graph_fraud_edge_count", "graph_fraud_edge_ratio"]
    if include_graph_mining:
        payer_cols = [*payer_cols, *mining_cols]
    for entity_col, prefix, cols in [
        ("payer_id", "payer", payer_cols),
        ("payee_id", "payee", ["graph_degree", *([] if not include_graph_mining else ["graph_mining_group_risk_score"])]),
        ("merchant_id", "merchant", ["graph_degree", *([] if not include_graph_mining else ["graph_mining_group_risk_score"])]),
        ("device_id", "device", ["graph_degree", *([] if not include_graph_mining else ["graph_mining_group_risk_score"])]),
        ("ip_id", "ip", ["graph_degree", *([] if not include_graph_mining else ["graph_mining_group_risk_score"])]),
    ]:
        if entity_col not in out.columns:
            continue
        available_cols = ["entity_id", *[col for col in cols if col in graph_features.columns]]
        selected = graph_features[available_cols].rename(columns={col: f"{prefix}_{col}" for col in available_cols if col != "entity_id"})
        out = out.merge(selected, left_on=entity_col, right_on="entity_id", how="left")
        out = out.drop(columns=["entity_id"])

    for col in expected_columns:
        if col not in out.columns:
            out[col] = None if col.endswith("_fraud_group_mining_id") else 0.0
        if not col.endswith("_fraud_group_mining_id"):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out
