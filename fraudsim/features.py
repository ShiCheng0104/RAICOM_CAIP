from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TARGET_COL = "is_fraud"

ID_AND_CONTROL_COLS = {
    TARGET_COL,
    "row_id",
    "transaction_id",
    "timestamp",
    "event_type",
    "split",
    "payer_id",
    "payee_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "user_id",
}

LEAKAGE_COLS = {
    "label_quality",
    "fraud_type",
    "risk_level",
    "rule_score",
    "fraud_txn_count",
    "observed_fraud_rate",
    "is_black_user",
    "risk_history_score",
    "fraud_txn_count_device",
    "observed_fraud_rate_device",
    "device_risk_score",
    "fraud_txn_count_ip",
    "observed_fraud_rate_ip",
    "ip_risk_score",
    "fraud_txn_count_merchant",
    "observed_fraud_rate_merchant",
    "merchant_risk_score",
    "is_high_risk_merchant",
}

TIME_COL_PATTERNS = ("first_seen_time", "last_seen_time")


@dataclass
class FeatureConfig:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_fill_values: dict[str, float]
    categorical_fill_value: str = "__missing__"
    target_col: str = TARGET_COL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureConfig":
        return cls(**data)


def _profile_path(dataset_dir: Path, name: str) -> Path:
    return dataset_dir / f"{name}.parquet"


def load_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir / "splits" / f"{split}.parquet")


def _drop_profile_leakage(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {}
    for col in renamed.columns:
        if col.endswith("_time") or col in {"first_seen_time", "last_seen_time"}:
            rename_map[col] = f"{prefix}_{col}"
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def enrich_transactions(tx: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    df = tx.copy()

    users = _drop_profile_leakage(pd.read_parquet(_profile_path(dataset_dir, "user_profile")), "user")
    devices = _drop_profile_leakage(pd.read_parquet(_profile_path(dataset_dir, "device_profile")), "device")
    ips = _drop_profile_leakage(pd.read_parquet(_profile_path(dataset_dir, "ip_geo_profile")), "ip")
    merchants = _drop_profile_leakage(pd.read_parquet(_profile_path(dataset_dir, "merchant_profile")), "merchant")

    df = df.merge(users, left_on="payer_id", right_on="user_id", how="left", suffixes=("", "_user"))
    df = df.merge(devices, on="device_id", how="left", suffixes=("", "_device"))
    df = df.merge(ips, on="ip_id", how="left", suffixes=("", "_ip"))
    df = df.merge(merchants, on="merchant_id", how="left", suffixes=("", "_merchant"))
    return df


def _is_time_col(col: str) -> bool:
    return any(pattern in col for pattern in TIME_COL_PATTERNS)


def select_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, FeatureConfig]:
    labeled = df[df[TARGET_COL].isin([0, 1])].copy()
    y = labeled[TARGET_COL].astype(int)

    drop_cols = set(ID_AND_CONTROL_COLS) | LEAKAGE_COLS
    drop_cols |= {col for col in labeled.columns if _is_time_col(col)}
    x = labeled.drop(columns=[col for col in drop_cols if col in labeled.columns])

    bool_cols = x.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        x[col] = x[col].astype("int8")

    categorical_columns = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [col for col in x.columns if col not in categorical_columns]
    numeric_fill_values = {}
    for col in numeric_columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        median = x[col].median()
        numeric_fill_values[col] = 0.0 if pd.isna(median) else float(median)
        x[col] = x[col].fillna(numeric_fill_values[col])

    for col in categorical_columns:
        x[col] = x[col].fillna("__missing__").astype("category")

    feature_config = FeatureConfig(
        feature_columns=x.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_fill_values=numeric_fill_values,
    )
    return x, y, feature_config


def apply_feature_config(df: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    x = df.copy()
    for col in feature_config.feature_columns:
        if col not in x.columns:
            x[col] = np.nan
    x = x[feature_config.feature_columns]

    for col in feature_config.feature_columns:
        if col in feature_config.categorical_columns:
            x[col] = x[col].fillna(feature_config.categorical_fill_value).astype("category")
        else:
            fill_value = feature_config.numeric_fill_values.get(col, 0.0)
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(fill_value)
    return x


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            if key == "user_profile":
                out.update(value)
            elif key == "device_profile":
                for sub_key, sub_value in value.items():
                    if sub_key in {"txn_count", "fraud_txn_count", "observed_fraud_rate"}:
                        out[f"{sub_key}_device"] = sub_value
                    elif sub_key in {"first_seen_time", "last_seen_time"}:
                        out[f"device_{sub_key}"] = sub_value
                    else:
                        out[sub_key] = sub_value
            elif key == "ip_profile":
                for sub_key, sub_value in value.items():
                    if sub_key in {"bind_user_count", "txn_count", "fraud_txn_count", "observed_fraud_rate"}:
                        out[f"{sub_key}_ip"] = sub_value
                    elif sub_key in {"first_seen_time", "last_seen_time"}:
                        out[f"ip_{sub_key}"] = sub_value
                    else:
                        out[sub_key] = sub_value
            elif key == "merchant_profile":
                for sub_key, sub_value in value.items():
                    if sub_key in {"merchant_category", "total_amount", "avg_amount", "fraud_txn_count", "observed_fraud_rate"}:
                        out[f"{sub_key}_merchant"] = sub_value
                    elif sub_key in {"txn_count"}:
                        out[f"{sub_key}_merchant"] = sub_value
                    elif sub_key in {"first_seen_time", "last_seen_time"}:
                        out[f"merchant_{sub_key}"] = sub_value
                    else:
                        out[sub_key] = sub_value
            elif key == "graph_features":
                out.update(value)
            else:
                for sub_key, sub_value in value.items():
                    out[f"{key}_{sub_key}"] = sub_value
        elif isinstance(value, str) and value and value[0] in "[{":
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                out[key] = value
            else:
                if isinstance(decoded, dict):
                    for sub_key, sub_value in decoded.items():
                        out[f"{key}_{sub_key}"] = sub_value
                else:
                    out[key] = value
        else:
            out[key] = value
    return out


def records_to_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([flatten_record(record) for record in records])
