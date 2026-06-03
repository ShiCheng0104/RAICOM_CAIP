"""
Validate the FP-FraudSim unified dataset.

Outputs:
    data/processed/fp_fraudsim/validation_report.json

Run:
    python scripts/etl/03_validate_unified_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_UNIFIED = ROOT / "data" / "processed" / "raw_unified"

DATASET = ROOT / "data" / "processed" / "fp_fraudsim"


def build_required_files(dataset: Path) -> dict[str, Path]:
    return {
        "transaction_log": dataset / "transaction_log.parquet",
        "user_profile": dataset / "user_profile.parquet",
        "merchant_profile": dataset / "merchant_profile.parquet",
        "device_profile": dataset / "device_profile.parquet",
        "ip_geo_profile": dataset / "ip_geo_profile.parquet",
        "graph_nodes": dataset / "graph_nodes.parquet",
        "graph_edges": dataset / "graph_edges.parquet",
        "transaction_stream": dataset / "transaction_stream.jsonl",
        "feature_store_index": dataset / "feature_store_index.parquet",
        "manifest": dataset / "manifest.json",
        "split_train": dataset / "splits" / "train.parquet",
        "split_valid": dataset / "splits" / "valid.parquet",
        "split_test": dataset / "splits" / "test.parquet",
        "split_unlabeled": dataset / "splits" / "unlabeled.parquet",
    }


REQUIRED_FILES = build_required_files(DATASET)

REQUIRED_TRANSACTION_COLS = {
    "transaction_id",
    "source",
    "timestamp",
    "amount",
    "currency",
    "txn_type",
    "channel",
    "payment_method",
    "payer_id",
    "payee_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "is_fraud",
    "fraud_type",
    "risk_level",
    "rule_score",
    "split",
}

REQUIRED_EDGE_COLS = {
    "src_id",
    "dst_id",
    "src_type",
    "dst_type",
    "edge_type",
    "weight",
    "timestamp",
    "source",
    "label",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def load_manifest() -> dict[str, Any]:
    path = REQUIRED_FILES["manifest"]
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_files() -> list[dict[str, Any]]:
    checks = []
    for name, path in REQUIRED_FILES.items():
        checks.append(check(
            f"file_exists:{name}",
            path.exists(),
            {"path": rel(path), "bytes": path.stat().st_size if path.exists() else 0},
        ))
    return checks


def validate_transactions(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    path = REQUIRED_FILES["transaction_log"]
    tx = pd.read_parquet(path)
    checks = []
    missing_cols = sorted(REQUIRED_TRANSACTION_COLS - set(tx.columns))
    checks.append(check("transaction_required_columns", not missing_cols, missing_cols))
    checks.append(check("transaction_id_unique", tx["transaction_id"].is_unique, {
        "rows": int(len(tx)),
        "unique_transaction_id": int(tx["transaction_id"].nunique()),
    }))
    checks.append(check("timestamp_not_null", tx["timestamp"].notna().all(), {
        "null_count": int(tx["timestamp"].isna().sum()),
    }))
    allowed_labels = {-1, 0, 1}
    labels = set(pd.to_numeric(tx["is_fraud"], errors="coerce").dropna().astype(int).unique())
    checks.append(check("is_fraud_values", labels <= allowed_labels, {
        "values": sorted(labels),
    }))
    checks.append(check("amount_non_negative", bool((tx["amount"] >= 0).all()), {
        "negative_count": int((tx["amount"] < 0).sum()),
    }))

    if manifest:
        expected = manifest.get("row_counts", {}).get("transaction_log")
        checks.append(check("manifest_transaction_count", expected == len(tx), {
            "manifest": expected,
            "actual": int(len(tx)),
        }))

    source_counts = tx.groupby("source").agg(
        rows=("transaction_id", "size"),
        fraud=("is_fraud", lambda s: int((s == 1).sum())),
        unlabeled=("is_fraud", lambda s: int((s < 0).sum())),
    ).reset_index()
    checks.append(check("transaction_sources_present", len(source_counts) >= 6, {
        row["source"]: {
            "rows": int(row["rows"]),
            "fraud": int(row["fraud"]),
            "unlabeled": int(row["unlabeled"]),
        }
        for _, row in source_counts.iterrows()
    }))
    return checks, tx


def validate_splits(tx: pd.DataFrame) -> list[dict[str, Any]]:
    checks = []
    split_parts = {}
    for split in ("train", "valid", "test", "unlabeled"):
        path = DATASET / "splits" / f"{split}.parquet"
        part = pd.read_parquet(path, columns=["transaction_id", "timestamp", "is_fraud", "split"])
        split_parts[split] = part
        checks.append(check(f"split_column:{split}", bool((part["split"] == split).all()), {
            "rows": int(len(part)),
            "bad_rows": int((part["split"] != split).sum()),
        }))

    split_total = sum(len(p) for p in split_parts.values())
    checks.append(check("split_total_matches_transaction_log", split_total == len(tx), {
        "split_total": int(split_total),
        "transaction_log": int(len(tx)),
    }))

    train_max = split_parts["train"]["timestamp"].max()
    valid_min = split_parts["valid"]["timestamp"].min()
    valid_max = split_parts["valid"]["timestamp"].max()
    test_min = split_parts["test"]["timestamp"].min()
    checks.append(check("labeled_time_order", train_max <= valid_min and valid_max <= test_min, {
        "train_max": str(train_max),
        "valid_min": str(valid_min),
        "valid_max": str(valid_max),
        "test_min": str(test_min),
    }))

    checks.append(check("unlabeled_labels", bool((split_parts["unlabeled"]["is_fraud"] == -1).all()), {
        "rows": int(len(split_parts["unlabeled"])),
        "bad_rows": int((split_parts["unlabeled"]["is_fraud"] != -1).sum()),
    }))
    return checks


def validate_profiles(tx: pd.DataFrame) -> list[dict[str, Any]]:
    checks = []
    user_ids = pd.read_parquet(REQUIRED_FILES["user_profile"], columns=["user_id"])["user_id"].astype(str)
    merchant_ids = pd.read_parquet(REQUIRED_FILES["merchant_profile"], columns=["merchant_id"])["merchant_id"].astype(str)
    device_ids = pd.read_parquet(REQUIRED_FILES["device_profile"], columns=["device_id"])["device_id"].astype(str)
    ip_ids = pd.read_parquet(REQUIRED_FILES["ip_geo_profile"], columns=["ip_id"])["ip_id"].astype(str)

    user_set = set(user_ids)
    merchant_set = set(merchant_ids)
    device_set = set(device_ids)
    ip_set = set(ip_ids)

    checks.append(check("payer_profile_coverage", tx["payer_id"].astype(str).isin(user_set).all(), {
        "missing_payers": int((~tx["payer_id"].astype(str).isin(user_set)).sum()),
    }))
    merchant_tx = tx[tx["merchant_id"].notna()]
    checks.append(check("merchant_profile_coverage", merchant_tx["merchant_id"].astype(str).isin(merchant_set).all(), {
        "merchant_transactions": int(len(merchant_tx)),
        "missing_merchants": int((~merchant_tx["merchant_id"].astype(str).isin(merchant_set)).sum()),
    }))
    checks.append(check("device_profile_coverage", tx["device_id"].astype(str).isin(device_set).all(), {
        "missing_devices": int((~tx["device_id"].astype(str).isin(device_set)).sum()),
    }))
    checks.append(check("ip_profile_coverage", tx["ip_id"].astype(str).isin(ip_set).all(), {
        "missing_ips": int((~tx["ip_id"].astype(str).isin(ip_set)).sum()),
    }))
    return checks


def validate_graph(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    nodes = pd.read_parquet(REQUIRED_FILES["graph_nodes"], columns=["node_id", "node_type", "label"])
    edges = pd.read_parquet(REQUIRED_FILES["graph_edges"])

    missing_edge_cols = sorted(REQUIRED_EDGE_COLS - set(edges.columns))
    checks.append(check("graph_edge_required_columns", not missing_edge_cols, missing_edge_cols))
    checks.append(check("graph_node_id_unique", nodes["node_id"].is_unique, {
        "rows": int(len(nodes)),
        "unique_node_id": int(nodes["node_id"].nunique()),
    }))

    node_set = set(nodes["node_id"].astype(str))
    missing_src = int((~edges["src_id"].astype(str).isin(node_set)).sum())
    missing_dst = int((~edges["dst_id"].astype(str).isin(node_set)).sum())
    checks.append(check("graph_edge_endpoint_coverage", missing_src == 0 and missing_dst == 0, {
        "missing_src": missing_src,
        "missing_dst": missing_dst,
    }))

    if manifest:
        expected_nodes = manifest.get("row_counts", {}).get("graph_nodes")
        expected_edges = manifest.get("row_counts", {}).get("graph_edges")
        checks.append(check("manifest_graph_counts", expected_nodes == len(nodes) and expected_edges == len(edges), {
            "manifest_nodes": expected_nodes,
            "actual_nodes": int(len(nodes)),
            "manifest_edges": expected_edges,
            "actual_edges": int(len(edges)),
        }))

    edge_types = edges["edge_type"].value_counts().to_dict()
    checks.append(check("graph_edge_types_present", len(edge_types) >= 5, {
        str(k): int(v) for k, v in edge_types.items()
    }))
    return checks


def validate_stream(sample_size: int) -> list[dict[str, Any]]:
    checks = []
    path = REQUIRED_FILES["transaction_stream"]
    rows = 0
    bad_rows = 0
    required = {"transaction_id", "source", "payer_id", "payee_id", "amount", "txn_type", "risk_level"}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if rows >= sample_size:
                break
            rows += 1
            try:
                event = json.loads(line)
                if not required <= set(event):
                    bad_rows += 1
            except json.JSONDecodeError:
                bad_rows += 1
    checks.append(check("transaction_stream_jsonl_sample", rows > 0 and bad_rows == 0, {
        "sampled_rows": rows,
        "bad_rows": bad_rows,
    }))
    return checks


def validate_feature_store() -> list[dict[str, Any]]:
    checks = []
    index = pd.read_parquet(REQUIRED_FILES["feature_store_index"])
    missing = []
    for _, row in index.iterrows():
        path = ROOT / str(row["path"])
        if not path.exists():
            # Older index values are relative to RAICOM_CAIP, but allow raw_unified fallback.
            alt = RAW_UNIFIED / Path(str(row["path"])).name
            if not alt.exists():
                missing.append(str(row["path"]))
    checks.append(check("feature_store_paths_exist", not missing, missing))
    checks.append(check("feature_store_has_join_keys", {"table", "path", "join_key"} <= set(index.columns), {
        "columns": list(index.columns),
        "rows": int(len(index)),
    }))
    return checks


def main() -> int:
    global DATASET, REQUIRED_FILES
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET),
                        help="Unified dataset directory to validate.")
    parser.add_argument("--stream-sample-size", type=int, default=10_000)
    args = parser.parse_args()

    DATASET = Path(args.dataset_dir)
    if not DATASET.is_absolute():
        DATASET = ROOT / DATASET
    REQUIRED_FILES = build_required_files(DATASET)

    checks: list[dict[str, Any]] = []
    checks.extend(validate_files())
    if not all(c["ok"] for c in checks):
        report = {"ok": False, "checks": checks}
        out = DATASET / "validation_report.json"
        write_json(out, report)
        print(f"validation failed: missing files; report={rel(out)}")
        return 1

    manifest = load_manifest()
    tx_checks, tx = validate_transactions(manifest)
    checks.extend(tx_checks)
    checks.extend(validate_splits(tx))
    checks.extend(validate_profiles(tx))
    checks.extend(validate_graph(manifest))
    checks.extend(validate_stream(args.stream_sample_size))
    checks.extend(validate_feature_store())

    report = {
        "ok": all(c["ok"] for c in checks),
        "dataset_dir": rel(DATASET),
        "summary": {
            "transaction_rows": int(len(tx)),
            "fraud_rows": int((tx["is_fraud"] == 1).sum()),
            "normal_rows": int((tx["is_fraud"] == 0).sum()),
            "unlabeled_rows": int((tx["is_fraud"] < 0).sum()),
            "sources": {str(k): int(v) for k, v in tx["source"].value_counts().to_dict().items()},
            "splits": {str(k): int(v) for k, v in tx["split"].value_counts().to_dict().items()},
            "time_min": str(tx["timestamp"].min()),
            "time_max": str(tx["timestamp"].max()),
        },
        "checks": checks,
    }
    out = DATASET / "validation_report.json"
    write_json(out, report)
    print(f"validation ok={report['ok']} report={rel(out)}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
