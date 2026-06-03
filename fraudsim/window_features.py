from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import pandas as pd


WINDOW_FEATURE_COLUMNS = [
    "window_features_user_txn_count_5min",
    "window_features_user_amount_sum_5min",
    "window_features_user_txn_count_1h",
    "window_features_user_amount_sum_1h",
    "window_features_user_unique_payee_count_1h",
    "window_features_device_unique_user_count_10min",
    "window_features_ip_unique_user_count_10min",
    "window_features_merchant_txn_count_1h",
    "window_features_merchant_amount_sum_1h",
    "window_features_merchant_unique_user_count_1h",
]


def parse_timestamp(value: Any) -> float:
    if value is None or value == "":
        return datetime.now(timezone.utc).timestamp()
    if isinstance(value, pd.Timestamp):
        return value.timestamp()
    if isinstance(value, datetime):
        return value.timestamp()
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        try:
            return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return datetime.now(timezone.utc).timestamp()


def _trim(events: deque[dict[str, Any]], now_ts: float, seconds: int) -> None:
    while events and now_ts - events[0]["ts"] > seconds:
        events.popleft()


def _sum_amount(events: deque[dict[str, Any]]) -> float:
    return float(sum(float(e.get("amount") or 0.0) for e in events))


def _unique(events: deque[dict[str, Any]], key: str) -> int:
    return len({e.get(key) for e in events if e.get(key)})


def compute_window_feature_dict(
    event: dict[str, Any],
    user_events: dict[str, deque[dict[str, Any]]],
    device_events: dict[str, deque[dict[str, Any]]],
    ip_events: dict[str, deque[dict[str, Any]]],
    merchant_events: dict[str, deque[dict[str, Any]]],
) -> dict[str, float | int]:
    now_ts = parse_timestamp(event.get("timestamp"))
    item = {
        "ts": now_ts,
        "amount": float(event.get("amount") or 0.0),
        "payer_id": event.get("payer_id"),
        "payee_id": event.get("payee_id"),
        "merchant_id": event.get("merchant_id"),
    }

    payer = str(event.get("payer_id") or "")
    device = str(event.get("device_id") or "")
    ip = str(event.get("ip_id") or "")
    merchant = str(event.get("merchant_id") or "")

    user_events[payer].append(item)
    device_events[device].append(item)
    ip_events[ip].append(item)
    merchant_events[merchant].append(item)

    _trim(user_events[payer], now_ts, 3600)
    _trim(device_events[device], now_ts, 600)
    _trim(ip_events[ip], now_ts, 600)
    _trim(merchant_events[merchant], now_ts, 3600)

    user_window = user_events[payer]
    user_window_5min = deque(e for e in user_window if now_ts - e["ts"] <= 300)
    device_window = device_events[device]
    ip_window = ip_events[ip]
    merchant_window = merchant_events[merchant]

    return {
        "user_txn_count_5min": len(user_window_5min),
        "user_amount_sum_5min": _sum_amount(user_window_5min),
        "user_txn_count_1h": len(user_window),
        "user_amount_sum_1h": _sum_amount(user_window),
        "user_unique_payee_count_1h": _unique(user_window, "payee_id"),
        "device_unique_user_count_10min": _unique(device_window, "payer_id"),
        "ip_unique_user_count_10min": _unique(ip_window, "payer_id"),
        "merchant_txn_count_1h": len(merchant_window),
        "merchant_amount_sum_1h": _sum_amount(merchant_window),
        "merchant_unique_user_count_1h": _unique(merchant_window, "payer_id"),
    }


def add_offline_window_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        out = df.copy()
        for col in WINDOW_FEATURE_COLUMNS:
            out[col] = 0.0
        return out

    out = df.copy()
    original_index = out.index
    sorted_df = out.sort_values("timestamp", kind="mergesort")

    user_events: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    device_events: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    ip_events: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    merchant_events: dict[str, deque[dict[str, Any]]] = defaultdict(deque)

    rows: list[dict[str, float | int]] = []
    for record in sorted_df.to_dict(orient="records"):
        features = compute_window_feature_dict(record, user_events, device_events, ip_events, merchant_events)
        rows.append({f"window_features_{key}": value for key, value in features.items()})

    feature_frame = pd.DataFrame(rows, index=sorted_df.index)
    out = sorted_df.join(feature_frame)
    return out.loc[original_index]
