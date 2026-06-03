"""
ETL Stage ② build a usable FP-FraudSim dataset.

Inputs:
    data/processed/raw_unified/*.parquet from 01_normalize.py
    data/banksim/bsNET140513_032310.csv
    data/elliptic/elliptic_bitcoin_dataset/*.csv
    data/dgraphfin/dgraphfin.npz
    data/amlsim/sample/outputs/*.csv

Outputs:
    data/processed/fp_fraudsim/transaction_log.parquet
    data/processed/fp_fraudsim/user_profile.parquet
    data/processed/fp_fraudsim/merchant_profile.parquet
    data/processed/fp_fraudsim/device_profile.parquet
    data/processed/fp_fraudsim/ip_geo_profile.parquet
    data/processed/fp_fraudsim/graph_nodes.parquet
    data/processed/fp_fraudsim/graph_edges.parquet
    data/processed/fp_fraudsim/transaction_stream.jsonl
    data/processed/fp_fraudsim/splits/{train,valid,test,unlabeled}.parquet
    data/processed/fp_fraudsim/manifest.json

The default mvp profile samples the large public datasets deterministically
while keeping enough fraud rows for demos and model training. Use --profile full
to consume all normalized transaction rows.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "processed" / "raw_unified"
OUT = DATA / "processed" / "fp_fraudsim"
SPLITS = OUT / "splits"
BASE_DATE = pd.Timestamp("2026-01-01 00:00:00")
END_DATE = pd.Timestamp("2026-03-31 23:59:59")
WINDOW_SECONDS = int((END_DATE - BASE_DATE).total_seconds())

MVP_CAPS = {
    "paysim": 200_000,
    "banksim": 120_000,
    "ieee_cis": 160_000,
    "creditcard": 80_000,
    "saml_d": 250_000,
}

CORE_SOURCES = ("paysim", "banksim", "ieee_cis", "creditcard", "saml_d")
CHANNELS = {
    "paysim": ("app", "web", "mini_program"),
    "banksim": ("pos", "qr_code", "app"),
    "ieee_cis": ("web", "app", "api"),
    "creditcard": ("pos", "web", "app"),
    "saml_d": ("bank_api", "web", "branch"),
    "amlsim_sample": ("core_banking", "branch", "batch"),
}
DEVICE_TYPES = ("mobile", "desktop", "tablet", "pos_terminal")
OS_TYPES = ("Android", "iOS", "Windows", "macOS", "Linux", "POS")
BROWSERS = ("Chrome", "Safari", "Edge", "Firefox", "WeChat WebView", "Bank Client")
ISPS = ("mobile", "telecom", "unicom", "cloud", "bank_network", "unknown")
CITY_BY_COUNTRY = {
    "SG": "Singapore",
    "ES": "Madrid",
    "EU": "Paris",
    "US": "New York",
    "UK": "London",
    "UAE": "Dubai",
    "unknown": "Unknown",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)


def stable_hash(values: pd.Series | pd.Index, salt: str = "") -> pd.Series:
    series = pd.Series(values, copy=False).astype(str)
    if salt:
        series = salt + ":" + series
    return pd.util.hash_pandas_object(series, index=False).astype("uint64")


def choose_by_hash(values: pd.Series, choices: tuple[str, ...], salt: str) -> pd.Series:
    h = stable_hash(values, salt=salt)
    idx = (h % len(choices)).astype("int64")
    return pd.Series(np.take(np.array(choices, dtype=object), idx), index=values.index)


def id_by_hash(values: pd.Series, prefix: str, modulo: int, salt: str) -> pd.Series:
    h = stable_hash(values, salt=salt)
    return prefix + (h % modulo).astype("int64").astype(str).str.zfill(7)


def parse_source_caps(raw_caps: list[str]) -> dict[str, int | None]:
    caps: dict[str, int | None] = {}
    for item in raw_caps:
        if "=" not in item:
            raise ValueError(f"Invalid --source-cap value: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip().lower()
        caps[name] = None if value in {"none", "full", "-1"} else int(value)
    return caps


def sample_with_label(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df.copy()

    rng_state = seed
    labeled_fraud = df[df["is_fraud"] == 1]
    labeled_normal = df[df["is_fraud"] == 0]
    unlabeled = df[df["is_fraud"] < 0]

    fraud_cap = min(len(labeled_fraud), max(1, int(cap * 0.25)))
    unknown_cap = min(len(unlabeled), int(cap * 0.10))
    normal_cap = max(0, cap - fraud_cap - unknown_cap)

    parts = []
    if fraud_cap:
        parts.append(labeled_fraud.sample(n=fraud_cap, random_state=rng_state))
    if normal_cap:
        parts.append(labeled_normal.sample(n=min(normal_cap, len(labeled_normal)),
                                           random_state=rng_state + 1))
    remaining = cap - sum(len(p) for p in parts)
    if unknown_cap or remaining > 0:
        n_unknown = min(len(unlabeled), max(unknown_cap, remaining))
        if n_unknown:
            parts.append(unlabeled.sample(n=n_unknown, random_state=rng_state + 2))

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=rng_state + 3).reset_index(drop=True)


def rescale_time(ts: pd.Series) -> pd.Series:
    values = pd.to_datetime(ts, errors="coerce")
    if values.isna().all():
        return pd.Series([BASE_DATE] * len(values), index=values.index)
    values = values.fillna(values.dropna().min())
    span = (values.max() - values.min()).total_seconds()
    if span <= 0:
        order = pd.Series(np.arange(len(values)), index=values.index)
        frac = order / max(len(values) - 1, 1)
    else:
        frac = (values - values.min()).dt.total_seconds() / span
    return BASE_DATE + pd.to_timedelta((frac * WINDOW_SECONDS).round().astype("int64"),
                                       unit="s")


def payment_method(txn_type: pd.Series) -> pd.Series:
    lower = txn_type.astype(str).str.lower()
    out = pd.Series("balance", index=txn_type.index, dtype=object)
    out[lower.str.contains("card|product|w|c|h|r", regex=True)] = "card"
    out[lower.str.contains("cash|withdrawal|deposit", regex=True)] = "cash"
    out[lower.str.contains("cross_border|transfer|wire|ach", regex=True)] = "bank_transfer"
    out[lower.str.contains("payment|online|topup|top_up", regex=True)] = "wallet"
    return out


def add_common_fields(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = rescale_time(out["timestamp"])
    out["source"] = source
    out["currency"] = out["currency"].fillna("UNKNOWN").astype(str)
    out["txn_type"] = out["txn_type"].fillna("unknown").astype(str).str.lower()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0).clip(lower=0)
    out["payer_id"] = out["payer_id"].fillna(f"{source}:unknown_payer").astype(str)
    out["payee_id"] = out["payee_id"].fillna(f"{source}:unknown_payee").astype(str)
    out["merchant_id"] = out["merchant_id"].where(out["merchant_id"].notna(), None)
    out["merchant_category"] = out["merchant_category"].where(
        out["merchant_category"].notna(), "unknown")
    out["payer_country"] = out["payer_country"].where(out["payer_country"].notna(), "unknown")
    out["payee_country"] = out["payee_country"].where(out["payee_country"].notna(), "unknown")
    out["is_fraud"] = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(-1).astype("int8")
    out["fraud_subtype"] = out["fraud_subtype"].where(out["fraud_subtype"].notna(), None)

    out["channel"] = choose_by_hash(out["transaction_id"], CHANNELS[source], "channel")
    out["payment_method"] = payment_method(out["txn_type"])
    out["device_id"] = id_by_hash(out["payer_id"] + out["transaction_id"],
                                  "D_", 280_000, "device")
    out["ip_id"] = id_by_hash(out["payer_id"] + out["timestamp"].astype(str),
                              "IP_", 180_000, "ip")
    out["event_type"] = "transaction"
    out["hour"] = out["timestamp"].dt.hour.astype("int8")
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype("int8")
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype("int8")
    out["is_night"] = out["hour"].between(0, 5).astype("int8")
    out["label_quality"] = np.select(
        [out["is_fraud"] == 1, out["is_fraud"] == 0],
        ["labeled_fraud", "labeled_normal"],
        default="unlabeled",
    )
    return out


def infer_fraud_type(df: pd.DataFrame) -> pd.Series:
    fraud = df["is_fraud"] == 1
    subtype = df["fraud_subtype"].fillna("").astype(str)
    out = pd.Series("normal", index=df.index, dtype=object)
    out[df["is_fraud"] < 0] = "unknown"
    out[fraud & (df["source"] == "creditcard")] = "card_fraud"
    out[fraud & (df["source"] == "ieee_cis")] = "ecommerce_fraud"
    out[fraud & (df["source"] == "banksim")] = "merchant_fraud"
    out[fraud & (df["source"] == "paysim")] = "payment_fraud"
    out[fraud & (df["source"] == "amlsim_sample")] = "amlsim_alert"
    aml_mask = fraud & (df["source"] == "saml_d")
    out[aml_mask] = subtype[aml_mask].str.lower().str.replace(r"\s+", "_", regex=True)
    laundering = fraud & df["txn_type"].str.contains("transfer|cross_border|wire|ach",
                                                     regex=True, na=False)
    out[laundering & out.isin(["payment_fraud", "normal"])] = "money_laundering"
    return out


def add_risk_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    amount_rank = out.groupby("source")["amount"].rank(pct=True).fillna(0)
    cross_country = (out["payer_country"].astype(str) != out["payee_country"].astype(str)).astype(int)
    high_risk_type = out["txn_type"].str.contains(
        "cash_out|withdrawal|cross_border|transfer|wire|ach", regex=True, na=False).astype(int)
    out["rule_score"] = (
        0.45 * amount_rank
        + 0.20 * out["is_night"].astype(float)
        + 0.20 * cross_country.astype(float)
        + 0.15 * high_risk_type.astype(float)
    ).clip(0, 1).round(4)
    out["fraud_type"] = infer_fraud_type(out)
    out["risk_level"] = np.select(
        [out["is_fraud"] == 1, out["is_fraud"] < 0, out["rule_score"] >= 0.70,
         out["rule_score"] >= 0.35],
        ["high", "unknown", "high", "medium"],
        default="low",
    )
    return out


def load_normalized_transactions(caps: dict[str, int | None], seed: int) -> pd.DataFrame:
    parts = []
    for i, source in enumerate(CORE_SOURCES):
        path = RAW / f"{source}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/etl/01_normalize.py first.")
        log(f"load {path.relative_to(ROOT)}")
        df = pd.read_parquet(path)
        df = sample_with_label(df, caps.get(source), seed + i * 10)
        df = add_common_fields(df, source)
        parts.append(df)
        log(f"  {source}: rows={len(df):,}, fraud={(df['is_fraud'] == 1).sum():,}, "
            f"unlabeled={(df['is_fraud'] < 0).sum():,}")
    amlsim = load_amlsim_sample()
    if not amlsim.empty:
        parts.append(add_common_fields(amlsim, "amlsim_sample"))
        log(f"  amlsim_sample: rows={len(amlsim):,}, fraud={(amlsim['is_fraud'] == 1).sum():,}")
    tx = pd.concat(parts, ignore_index=True)
    tx = add_risk_fields(tx)
    tx = tx.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    tx.insert(0, "row_id", np.arange(len(tx), dtype=np.int64))
    return tx


def load_amlsim_sample() -> pd.DataFrame:
    sample = DATA / "amlsim" / "sample" / "outputs"
    tx_path = sample / "tx.csv"
    cash_path = sample / "cash_tx.csv"
    alerts_path = sample / "alerts.csv"
    if not tx_path.exists() or not cash_path.exists():
        return pd.DataFrame()
    alert_accounts: set[str] = set()
    if alerts_path.exists():
        alerts = pd.read_csv(alerts_path)
        alert_accounts = set(alerts["ACCOUNT_ID"].astype(str))

    parts = []
    tx = pd.read_csv(tx_path)
    if not tx.empty:
        payer = tx["ACCOUNT_ID"].astype(str)
        payee = tx["COUNTER_PARTY_ACCOUNT_NUM"].astype(str)
        parts.append(pd.DataFrame({
            "transaction_id": "amlsim_tx:" + tx["TXN_ID"].astype(str),
            "source": "amlsim_sample",
            "timestamp": BASE_DATE + pd.to_timedelta(tx["start"].astype(int), unit="D"),
            "amount": tx["TXN_AMOUNT_ORIG"].astype(float),
            "currency": "USD",
            "txn_type": tx["TXN_SOURCE_TYPE_CODE"].astype(str).str.lower(),
            "payer_id": "amlsim_a:" + payer,
            "payee_id": "amlsim_a:" + payee,
            "merchant_id": None,
            "merchant_category": "aml_transfer",
            "payer_country": "US",
            "payee_country": "US",
            "is_fraud": (payer.isin(alert_accounts) | payee.isin(alert_accounts)).astype("int8"),
            "fraud_subtype": np.where(
                payer.isin(alert_accounts) | payee.isin(alert_accounts), "cycle", None),
        }))
    cash = pd.read_csv(cash_path)
    if not cash.empty:
        payer = cash["ACCOUNT_ID"].astype(str)
        branch = cash["BRANCH_ID"].astype(str)
        parts.append(pd.DataFrame({
            "transaction_id": "amlsim_cash:" + cash["TXN_ID"].astype(str),
            "source": "amlsim_sample",
            "timestamp": BASE_DATE + pd.to_timedelta(cash["RUN_DATE"].astype(int), unit="D"),
            "amount": cash["TXN_AMOUNT_ORIG"].astype(float),
            "currency": "USD",
            "txn_type": cash["TXN_SOURCE_TYPE_CODE"].astype(str).str.lower(),
            "payer_id": "amlsim_a:" + payer,
            "payee_id": "amlsim_branch:" + branch,
            "merchant_id": "amlsim_branch:" + branch,
            "merchant_category": "cash_service",
            "payer_country": "US",
            "payee_country": "US",
            "is_fraud": payer.isin(alert_accounts).astype("int8"),
            "fraud_subtype": np.where(payer.isin(alert_accounts), "cycle", None),
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def assign_time_splits(tx: pd.DataFrame) -> pd.DataFrame:
    out = tx.copy()
    out["split"] = "unlabeled"
    labeled_idx = out.index[out["is_fraud"].isin([0, 1])]
    n = len(labeled_idx)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    out.loc[labeled_idx[:train_end], "split"] = "train"
    out.loc[labeled_idx[train_end:valid_end], "split"] = "valid"
    out.loc[labeled_idx[valid_end:], "split"] = "test"
    return out


def amount_bucket(amount: pd.Series) -> pd.Series:
    bins = [-math.inf, 10, 100, 1_000, 10_000, math.inf]
    labels = ["micro", "small", "medium", "large", "xlarge"]
    return pd.cut(amount, bins=bins, labels=labels).astype(str)


def finalize_transaction_columns(tx: pd.DataFrame) -> pd.DataFrame:
    out = tx.copy()
    out["amount_bucket"] = amount_bucket(out["amount"])
    cols = [
        "row_id", "transaction_id", "source", "timestamp", "event_type",
        "amount", "amount_bucket", "currency", "txn_type", "channel",
        "payment_method", "payer_id", "payee_id", "merchant_id",
        "merchant_category", "device_id", "ip_id", "payer_country",
        "payee_country", "is_fraud", "fraud_type", "risk_level",
        "rule_score", "label_quality", "split", "hour", "day_of_week",
        "is_weekend", "is_night",
    ]
    return out[cols]


def build_user_profile(tx: pd.DataFrame) -> pd.DataFrame:
    payer = tx[["payer_id", "timestamp", "amount", "is_fraud", "device_id", "ip_id",
                "payer_country", "source"]].rename(columns={
                    "payer_id": "user_id",
                    "payer_country": "country",
                })
    payer["direction"] = "out"
    payee = tx[tx["merchant_id"].isna()][["payee_id", "timestamp", "amount", "is_fraud",
                                            "device_id", "ip_id", "payee_country", "source"]]
    payee = payee.rename(columns={"payee_id": "user_id", "payee_country": "country"})
    payee["direction"] = "in"
    events = pd.concat([payer, payee], ignore_index=True)

    g = events.groupby("user_id", sort=False)
    profile = g.agg(
        first_seen_time=("timestamp", "min"),
        last_seen_time=("timestamp", "max"),
        txn_count=("amount", "size"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        fraud_txn_count=("is_fraud", lambda s: int((s == 1).sum())),
        common_device_count=("device_id", "nunique"),
        common_ip_count=("ip_id", "nunique"),
        source_hint=("source", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        home_country=("country", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
    ).reset_index()
    profile["observed_fraud_rate"] = (
        profile["fraud_txn_count"] / profile["txn_count"].replace(0, np.nan)
    ).fillna(0).round(6)
    profile["account_age_days"] = (
        stable_hash(profile["user_id"], "account_age") % 1_800 + 1
    ).astype("int32")
    amount_score = profile["avg_amount"].rank(pct=True).fillna(0)
    profile["risk_history_score"] = (
        0.75 * profile["observed_fraud_rate"] + 0.25 * amount_score
    ).clip(0, 1).round(4)
    profile["is_black_user"] = (
        (profile["fraud_txn_count"] > 0) & (profile["risk_history_score"] >= 0.20)
    ).astype("int8")
    return profile


def build_merchant_profile(tx: pd.DataFrame) -> pd.DataFrame:
    src = tx[tx["merchant_id"].notna()].copy()
    if src.empty:
        return pd.DataFrame(columns=[
            "merchant_id", "merchant_category", "merchant_country", "first_seen_time",
            "last_seen_time", "txn_count", "unique_user_count", "total_amount",
            "avg_amount", "fraud_txn_count", "observed_fraud_rate",
            "chargeback_rate", "complaint_rate", "merchant_risk_score",
            "is_high_risk_merchant",
        ])
    g = src.groupby("merchant_id", sort=False)
    profile = g.agg(
        merchant_category=("merchant_category", lambda s: s.mode().iat[0]
                           if not s.mode().empty else "unknown"),
        merchant_country=("payee_country", lambda s: s.mode().iat[0]
                          if not s.mode().empty else "unknown"),
        first_seen_time=("timestamp", "min"),
        last_seen_time=("timestamp", "max"),
        txn_count=("transaction_id", "size"),
        unique_user_count=("payer_id", "nunique"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        fraud_txn_count=("is_fraud", lambda s: int((s == 1).sum())),
    ).reset_index()
    profile["observed_fraud_rate"] = (
        profile["fraud_txn_count"] / profile["txn_count"].replace(0, np.nan)
    ).fillna(0).round(6)
    h = stable_hash(profile["merchant_id"], "merchant_noise")
    profile["chargeback_rate"] = (
        profile["observed_fraud_rate"] * 0.6 + ((h % 100) / 10_000)
    ).clip(0, 1).round(5)
    profile["complaint_rate"] = (
        profile["observed_fraud_rate"] * 0.4 + (((h // 100) % 100) / 12_000)
    ).clip(0, 1).round(5)
    amount_score = profile["avg_amount"].rank(pct=True).fillna(0)
    profile["merchant_risk_score"] = (
        0.55 * profile["observed_fraud_rate"] + 0.20 * profile["chargeback_rate"]
        + 0.15 * profile["complaint_rate"] + 0.10 * amount_score
    ).clip(0, 1).round(4)
    profile["is_high_risk_merchant"] = (profile["merchant_risk_score"] >= 0.25).astype("int8")
    return profile


def build_device_profile(tx: pd.DataFrame) -> pd.DataFrame:
    g = tx.groupby("device_id", sort=False)
    profile = g.agg(
        first_seen_time=("timestamp", "min"),
        last_seen_time=("timestamp", "max"),
        bind_user_count=("payer_id", "nunique"),
        txn_count=("transaction_id", "size"),
        fraud_txn_count=("is_fraud", lambda s: int((s == 1).sum())),
    ).reset_index()
    h = stable_hash(profile["device_id"], "device_profile")
    profile["device_type"] = np.take(np.array(DEVICE_TYPES, dtype=object),
                                     (h % len(DEVICE_TYPES)).astype("int64"))
    profile["os"] = np.take(np.array(OS_TYPES, dtype=object),
                            ((h // 7) % len(OS_TYPES)).astype("int64"))
    profile["browser"] = np.take(np.array(BROWSERS, dtype=object),
                                 ((h // 13) % len(BROWSERS)).astype("int64"))
    profile["observed_fraud_rate"] = (
        profile["fraud_txn_count"] / profile["txn_count"].replace(0, np.nan)
    ).fillna(0)
    bind_score = (profile["bind_user_count"].rank(pct=True) * 0.45).fillna(0)
    profile["is_emulator"] = ((h % 97) < 3).astype("int8")
    profile["is_proxy_device"] = ((profile["bind_user_count"] >= 8) | ((h % 211) == 0)).astype("int8")
    profile["device_risk_score"] = (
        bind_score + 0.45 * profile["observed_fraud_rate"]
        + 0.05 * profile["is_emulator"] + 0.05 * profile["is_proxy_device"]
    ).clip(0, 1).round(4)
    return profile


def fake_ip(h: pd.Series) -> pd.Series:
    a = 10 + (h % 210).astype("int64")
    b = ((h // 210) % 256).astype("int64")
    c = ((h // (210 * 256)) % 256).astype("int64")
    d = ((h // (210 * 256 * 256)) % 254 + 1).astype("int64")
    return a.astype(str) + "." + b.astype(str) + "." + c.astype(str) + "." + d.astype(str)


def build_ip_profile(tx: pd.DataFrame) -> pd.DataFrame:
    g = tx.groupby("ip_id", sort=False)
    profile = g.agg(
        first_seen_time=("timestamp", "min"),
        last_seen_time=("timestamp", "max"),
        country=("payer_country", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        bind_user_count=("payer_id", "nunique"),
        txn_count=("transaction_id", "size"),
        fraud_txn_count=("is_fraud", lambda s: int((s == 1).sum())),
    ).reset_index()
    h = stable_hash(profile["ip_id"], "ip_profile")
    profile["ip_address"] = fake_ip(h)
    profile["city"] = profile["country"].map(CITY_BY_COUNTRY).fillna("Unknown")
    profile["isp"] = np.take(np.array(ISPS, dtype=object), (h % len(ISPS)).astype("int64"))
    profile["is_proxy"] = ((profile["bind_user_count"] >= 10) | ((h % 101) < 4)).astype("int8")
    profile["is_vpn"] = ((h % 127) < 5).astype("int8")
    profile["observed_fraud_rate"] = (
        profile["fraud_txn_count"] / profile["txn_count"].replace(0, np.nan)
    ).fillna(0)
    bind_score = (profile["bind_user_count"].rank(pct=True) * 0.35).fillna(0)
    profile["ip_risk_score"] = (
        bind_score + 0.45 * profile["observed_fraud_rate"]
        + 0.10 * profile["is_proxy"] + 0.10 * profile["is_vpn"]
    ).clip(0, 1).round(4)
    return profile


def edge_frame(src: pd.Series, dst: pd.Series, src_type: str, dst_type: str,
               edge_type: str, timestamp: pd.Series | pd.Timestamp | None,
               source: str, weight: pd.Series | float = 1.0,
               label: pd.Series | int | None = None) -> pd.DataFrame:
    n = len(src)
    if isinstance(weight, pd.Series):
        weight_values = weight.astype(float).values
    else:
        weight_values = np.full(n, float(weight))
    if timestamp is None:
        ts_values = pd.Series([pd.NaT] * n)
    elif isinstance(timestamp, pd.Series):
        ts_values = timestamp.reset_index(drop=True)
    else:
        ts_values = pd.Series([timestamp] * n)
    if label is None:
        label_values = np.full(n, -1, dtype=np.int8)
    elif isinstance(label, pd.Series):
        label_values = label.astype("int8").values
    else:
        label_values = np.full(n, int(label), dtype=np.int8)
    return pd.DataFrame({
        "src_id": src.astype(str).values,
        "dst_id": dst.astype(str).values,
        "src_type": src_type,
        "dst_type": dst_type,
        "edge_type": edge_type,
        "weight": weight_values,
        "timestamp": ts_values,
        "source": source,
        "label": label_values,
    })


def build_transaction_edges(tx: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    src = tx
    if cap is not None and len(src) > cap:
        src = src.sample(n=cap, random_state=seed).sort_values("timestamp")
    parts = [
        edge_frame(src["payer_id"], src["transaction_id"], "user", "transaction",
                   "make_transaction", src["timestamp"], "transaction_log", 1.0,
                   src["is_fraud"].clip(lower=0)),
        edge_frame(src["transaction_id"], src["payee_id"], "transaction", "account",
                   "transfer_to", src["timestamp"], "transaction_log", src["amount"],
                   src["is_fraud"].clip(lower=0)),
        edge_frame(src["payer_id"], src["device_id"], "user", "device",
                   "uses_device", src["timestamp"], "transaction_log", 1.0,
                   src["is_fraud"].clip(lower=0)),
        edge_frame(src["payer_id"], src["ip_id"], "user", "ip",
                   "login_ip", src["timestamp"], "transaction_log", 1.0,
                   src["is_fraud"].clip(lower=0)),
    ]
    merchant = src[src["merchant_id"].notna()]
    if not merchant.empty:
        parts.append(edge_frame(merchant["transaction_id"], merchant["merchant_id"],
                                "transaction", "merchant", "paid_to_merchant",
                                merchant["timestamp"], "transaction_log",
                                merchant["amount"], merchant["is_fraud"].clip(lower=0)))
    return pd.concat(parts, ignore_index=True)


def load_banksim_network(cap: int | None, seed: int) -> pd.DataFrame:
    path = DATA / "banksim" / "bsNET140513_032310.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if cap is not None and len(df) > cap:
        df = df.sample(n=cap, random_state=seed)
    src = "banksim_u:" + df["Source"].astype(str).str.strip("'")
    dst = "banksim_m:" + df["Target"].astype(str).str.strip("'")
    return edge_frame(src, dst, "user", "merchant", "banksim_payment_network",
                      BASE_DATE, "banksim_net", df["Weight"].astype(float),
                      df["fraud"].astype("int8"))


def load_elliptic_graph(edge_cap: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = DATA / "elliptic" / "elliptic_bitcoin_dataset"
    classes_path = base / "elliptic_txs_classes.csv"
    features_path = base / "elliptic_txs_features.csv"
    edges_path = base / "elliptic_txs_edgelist.csv"
    if not classes_path.exists() or not edges_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    classes = pd.read_csv(classes_path)
    classes["node_id"] = "elliptic_tx:" + classes["txId"].astype(str)
    classes["node_type"] = "crypto_transaction"
    classes["source"] = "elliptic"
    classes["label"] = np.select(
        [classes["class"].astype(str) == "1", classes["class"].astype(str) == "2"],
        [1, 0],
        default=-1,
    ).astype("int8")
    classes["raw_label"] = classes["class"].astype(str)
    if features_path.exists():
        time_steps = pd.read_csv(features_path, header=None, usecols=[0, 1],
                                 names=["txId", "time_step"])
        classes = classes.merge(time_steps, on="txId", how="left")
    else:
        classes["time_step"] = pd.NA
    classes["timestamp"] = BASE_DATE + pd.to_timedelta(
        pd.to_numeric(classes["time_step"], errors="coerce").fillna(0).astype(int),
        unit="D",
    )
    nodes = classes[["node_id", "node_type", "source", "label", "raw_label",
                     "time_step", "timestamp"]]

    edges = pd.read_csv(edges_path)
    if edge_cap is not None and len(edges) > edge_cap:
        edges = edges.sample(n=edge_cap, random_state=seed)
    src = "elliptic_tx:" + edges["txId1"].astype(str)
    dst = "elliptic_tx:" + edges["txId2"].astype(str)
    edge_ts = src.map(nodes.set_index("node_id")["timestamp"]).reset_index(drop=True)
    graph_edges = edge_frame(src, dst, "crypto_transaction", "crypto_transaction",
                             "elliptic_money_flow", edge_ts, "elliptic", 1.0)
    return nodes, graph_edges


def load_dgraphfin_graph(edge_cap: int | None, node_feature_cap: int | None,
                         seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = DATA / "dgraphfin" / "dgraphfin.npz"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    npz = np.load(path, mmap_mode="r")
    y = npz["y"]
    edge_index = npz["edge_index"]
    edge_type = npz["edge_type"]
    edge_timestamp = npz["edge_timestamp"]

    rng = np.random.default_rng(seed)
    total_edges = edge_index.shape[0]
    if edge_cap is not None and total_edges > edge_cap:
        edge_rows = np.sort(rng.choice(total_edges, size=edge_cap, replace=False))
    else:
        edge_rows = np.arange(total_edges)

    sampled_edges = edge_index[edge_rows]
    sampled_type = edge_type[edge_rows]
    sampled_ts = edge_timestamp[edge_rows]
    src = pd.Series(sampled_edges[:, 0].astype(str)).radd("dgraph_u:")
    dst = pd.Series(sampled_edges[:, 1].astype(str)).radd("dgraph_u:")
    ts_min = sampled_ts.min() if len(sampled_ts) else 0
    ts_span = max(int(sampled_ts.max() - ts_min), 1) if len(sampled_ts) else 1
    timestamps = BASE_DATE + pd.to_timedelta(
        ((sampled_ts - ts_min) / ts_span * WINDOW_SECONDS).round().astype("int64"),
        unit="s",
    )
    graph_edges = edge_frame(src, dst, "user", "user", "dgraph_relation",
                             pd.Series(timestamps), "dgraphfin", 1.0)
    graph_edges["edge_subtype"] = sampled_type.astype("int16")

    node_ids = np.unique(sampled_edges.reshape(-1))
    raw_labels = y[node_ids]
    labels = np.select([raw_labels == 1, raw_labels == 0], [1, 0], default=-1).astype("int8")
    nodes = pd.DataFrame({
        "node_id": pd.Series(node_ids.astype(str)).radd("dgraph_u:").values,
        "node_type": "user",
        "source": "dgraphfin",
        "label": labels,
        "raw_label": raw_labels.astype(str),
        "time_step": pd.NA,
        "timestamp": pd.NaT,
    })

    features = pd.DataFrame()
    if node_feature_cap:
        feature_nodes = node_ids[:min(node_feature_cap, len(node_ids))]
        x = npz["x"][feature_nodes]
        features = pd.DataFrame(x, columns=[f"dgraph_x{i}" for i in range(x.shape[1])])
        features.insert(0, "node_id",
                        pd.Series(feature_nodes.astype(str)).radd("dgraph_u:").values)
    return nodes, graph_edges, features


def load_amlsim_graph() -> pd.DataFrame:
    tx = load_amlsim_sample()
    if tx.empty:
        return pd.DataFrame()
    return edge_frame(tx["payer_id"], tx["payee_id"], "user", "account",
                      "amlsim_transfer", tx["timestamp"], "amlsim_sample",
                      tx["amount"], tx["is_fraud"])


def build_graph(tx: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("build graph edges")
    edge_parts = [
        build_transaction_edges(tx, args.transaction_edge_cap, args.seed),
        load_banksim_network(args.banksim_edge_cap, args.seed + 1),
        load_amlsim_graph(),
    ]
    elliptic_nodes, elliptic_edges = load_elliptic_graph(args.elliptic_edge_cap, args.seed + 2)
    dgraph_nodes, dgraph_edges, dgraph_features = load_dgraphfin_graph(
        args.dgraph_edge_cap, args.dgraph_node_feature_cap, args.seed + 3)
    edge_parts.extend([elliptic_edges, dgraph_edges])
    graph_edges = pd.concat([p for p in edge_parts if not p.empty], ignore_index=True)
    if "edge_subtype" not in graph_edges.columns:
        graph_edges["edge_subtype"] = pd.NA
    else:
        graph_edges["edge_subtype"] = graph_edges["edge_subtype"].astype("Int16")

    entity_nodes = build_entity_nodes(tx)
    endpoint_nodes = build_endpoint_nodes(graph_edges)
    node_parts = [entity_nodes, endpoint_nodes, elliptic_nodes, dgraph_nodes]
    graph_nodes = pd.concat([p for p in node_parts if not p.empty], ignore_index=True)
    graph_nodes = graph_nodes.drop_duplicates("node_id").reset_index(drop=True)

    if not dgraph_features.empty:
        dgraph_features.to_parquet(OUT / "dgraphfin_node_features_sample.parquet",
                                   index=False, compression="snappy")
    return graph_nodes, graph_edges


def build_entity_nodes(tx: pd.DataFrame) -> pd.DataFrame:
    user_ids = pd.concat([tx["payer_id"], tx.loc[tx["merchant_id"].isna(), "payee_id"]],
                         ignore_index=True).dropna().astype(str).drop_duplicates()
    transaction_ids = tx["transaction_id"].dropna().astype(str).drop_duplicates()
    merchant_ids = tx["merchant_id"].dropna().astype(str).drop_duplicates()
    device_ids = tx["device_id"].dropna().astype(str).drop_duplicates()
    ip_ids = tx["ip_id"].dropna().astype(str).drop_duplicates()

    parts = []
    for ids, node_type in ((user_ids, "user"), (transaction_ids, "transaction"),
                           (merchant_ids, "merchant"), (device_ids, "device"),
                           (ip_ids, "ip")):
        if ids.empty:
            continue
        parts.append(pd.DataFrame({
            "node_id": ids.values,
            "node_type": node_type,
            "source": "fp_fraudsim",
            "label": -1,
            "raw_label": "entity",
            "time_step": pd.NA,
            "timestamp": pd.NaT,
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_endpoint_nodes(graph_edges: pd.DataFrame) -> pd.DataFrame:
    src = graph_edges[["src_id", "src_type", "source"]].rename(
        columns={"src_id": "node_id", "src_type": "node_type"})
    dst = graph_edges[["dst_id", "dst_type", "source"]].rename(
        columns={"dst_id": "node_id", "dst_type": "node_type"})
    nodes = pd.concat([src, dst], ignore_index=True)
    nodes = nodes.dropna(subset=["node_id"]).drop_duplicates("node_id")
    nodes["label"] = -1
    nodes["raw_label"] = "edge_endpoint"
    nodes["time_step"] = pd.NA
    nodes["timestamp"] = pd.NaT
    return nodes[["node_id", "node_type", "source", "label", "raw_label",
                  "time_step", "timestamp"]]


def write_outputs(tx: pd.DataFrame, user_profile: pd.DataFrame,
                  merchant_profile: pd.DataFrame, device_profile: pd.DataFrame,
                  ip_profile: pd.DataFrame, graph_nodes: pd.DataFrame,
                  graph_edges: pd.DataFrame, stream_limit: int) -> dict[str, Any]:
    log("write unified dataset")
    outputs = {
        "transaction_log": OUT / "transaction_log.parquet",
        "user_profile": OUT / "user_profile.parquet",
        "merchant_profile": OUT / "merchant_profile.parquet",
        "device_profile": OUT / "device_profile.parquet",
        "ip_geo_profile": OUT / "ip_geo_profile.parquet",
        "graph_nodes": OUT / "graph_nodes.parquet",
        "graph_edges": OUT / "graph_edges.parquet",
    }
    tx.to_parquet(outputs["transaction_log"], index=False, compression="snappy")
    user_profile.to_parquet(outputs["user_profile"], index=False, compression="snappy")
    merchant_profile.to_parquet(outputs["merchant_profile"], index=False, compression="snappy")
    device_profile.to_parquet(outputs["device_profile"], index=False, compression="snappy")
    ip_profile.to_parquet(outputs["ip_geo_profile"], index=False, compression="snappy")
    graph_nodes.to_parquet(outputs["graph_nodes"], index=False, compression="snappy")
    graph_edges.to_parquet(outputs["graph_edges"], index=False, compression="snappy")

    split_paths = {}
    for split in ("train", "valid", "test", "unlabeled"):
        part = tx[tx["split"] == split]
        path = SPLITS / f"{split}.parquet"
        part.to_parquet(path, index=False, compression="snappy")
        split_paths[split] = path

    stream_cols = [
        "timestamp", "transaction_id", "source", "payer_id", "payee_id",
        "merchant_id", "amount", "currency", "txn_type", "channel",
        "payment_method", "device_id", "ip_id", "is_fraud", "fraud_type",
        "risk_level", "rule_score",
    ]
    stream = tx.sort_values("timestamp").head(stream_limit)
    stream_path = OUT / "transaction_stream.jsonl"
    stream[stream_cols].to_json(stream_path, orient="records", lines=True,
                                force_ascii=False, date_format="iso")

    feature_index = pd.DataFrame([
        {
            "table": "paysim_features",
            "path": "data/processed/raw_unified/paysim_features.parquet",
            "join_key": "transaction_id",
            "description": "PaySim balance before/after fields.",
        },
        {
            "table": "banksim_features",
            "path": "data/processed/raw_unified/banksim_features.parquet",
            "join_key": "transaction_id",
            "description": "BankSim user age, gender, and zipcode fields.",
        },
        {
            "table": "ieee_cis_features",
            "path": "data/processed/raw_unified/ieee_cis_features.parquet",
            "join_key": "transaction_id",
            "description": "IEEE-CIS high-dimensional transaction, card, email, device, and identity fields.",
        },
        {
            "table": "creditcard_features",
            "path": "data/processed/raw_unified/creditcard_features.parquet",
            "join_key": "transaction_id",
            "description": "CreditCard PCA features V1-V28.",
        },
        {
            "table": "dgraphfin_node_features_sample",
            "path": "data/processed/fp_fraudsim/dgraphfin_node_features_sample.parquet",
            "join_key": "node_id",
            "description": "Sampled DGraph-Fin 17-dimensional node features.",
        },
    ])
    feature_index.to_parquet(OUT / "feature_store_index.parquet",
                             index=False, compression="snappy")

    return {
        "files": {k: str(v.relative_to(ROOT)) for k, v in outputs.items()},
        "splits": {k: str(v.relative_to(ROOT)) for k, v in split_paths.items()},
        "stream": str(stream_path.relative_to(ROOT)),
        "feature_store_index": str((OUT / "feature_store_index.parquet").relative_to(ROOT)),
    }


def write_manifest(tx: pd.DataFrame, graph_nodes: pd.DataFrame,
                   graph_edges: pd.DataFrame, files: dict[str, Any],
                   args: argparse.Namespace) -> None:
    counts_by_source = tx.groupby("source").agg(
        rows=("transaction_id", "size"),
        fraud=("is_fraud", lambda s: int((s == 1).sum())),
        unlabeled=("is_fraud", lambda s: int((s < 0).sum())),
        min_time=("timestamp", "min"),
        max_time=("timestamp", "max"),
    ).reset_index()
    counts_by_source["min_time"] = counts_by_source["min_time"].astype(str)
    counts_by_source["max_time"] = counts_by_source["max_time"].astype(str)
    split_counts = tx["split"].value_counts().to_dict()
    manifest = {
        "dataset": "FP-FraudSim",
        "created_at": pd.Timestamp.now().isoformat(),
        "profile": args.profile,
        "seed": args.seed,
        "description": "Unified multi-source financial anti-fraud simulation dataset.",
        "time_window": {
            "start": str(tx["timestamp"].min()),
            "end": str(tx["timestamp"].max()),
            "split_policy": "labeled rows are sorted by timestamp: 70% train, 15% valid, 15% test; unlabeled IEEE-CIS test rows are written separately.",
        },
        "row_counts": {
            "transaction_log": int(len(tx)),
            "user_profile": int(pd.read_parquet(OUT / "user_profile.parquet", columns=["user_id"]).shape[0]),
            "merchant_profile": int(pd.read_parquet(OUT / "merchant_profile.parquet", columns=["merchant_id"]).shape[0]),
            "device_profile": int(pd.read_parquet(OUT / "device_profile.parquet", columns=["device_id"]).shape[0]),
            "ip_geo_profile": int(pd.read_parquet(OUT / "ip_geo_profile.parquet", columns=["ip_id"]).shape[0]),
            "graph_nodes": int(len(graph_nodes)),
            "graph_edges": int(len(graph_edges)),
        },
        "splits": {k: int(v) for k, v in split_counts.items()},
        "fraud": {
            "labeled_fraud_rows": int((tx["is_fraud"] == 1).sum()),
            "labeled_normal_rows": int((tx["is_fraud"] == 0).sum()),
            "unlabeled_rows": int((tx["is_fraud"] < 0).sum()),
            "fraud_rate_labeled": float(
                (tx["is_fraud"] == 1).sum() / max(int(tx["is_fraud"].isin([0, 1]).sum()), 1)
            ),
        },
        "sources": counts_by_source.to_dict(orient="records"),
        "files": files,
        "notes": [
            "Use transaction_id to join source-specific high-dimensional feature tables.",
            "Use graph_edges/graph_nodes for group fraud, AML path tracing, and GNN experiments.",
            "MVP profile is a deterministic sample, not the original population distribution.",
            "Run with --profile full when enough disk and memory are available.",
        ],
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"manifest -> {manifest_path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["mvp", "full"], default="mvp",
                        help="mvp samples large sources; full uses all normalized transaction rows.")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--source-cap", action="append", default=[],
                        help="Override one cap, for example --source-cap paysim=500000. Use none/full/-1 for all rows.")
    parser.add_argument("--transaction-edge-cap", type=int, default=250_000,
                        help="Number of transaction rows used to derive entity graph edges; -1 for all.")
    parser.add_argument("--banksim-edge-cap", type=int, default=200_000)
    parser.add_argument("--elliptic-edge-cap", type=int, default=234_355)
    parser.add_argument("--dgraph-edge-cap", type=int, default=300_000)
    parser.add_argument("--dgraph-node-feature-cap", type=int, default=100_000)
    parser.add_argument("--stream-limit", type=int, default=100_000)
    args = parser.parse_args()

    ensure_dirs()
    caps = {s: None for s in CORE_SOURCES} if args.profile == "full" else MVP_CAPS.copy()
    caps.update(parse_source_caps(args.source_cap))
    if args.transaction_edge_cap < 0:
        args.transaction_edge_cap = None
    if args.banksim_edge_cap < 0:
        args.banksim_edge_cap = None
    if args.elliptic_edge_cap < 0:
        args.elliptic_edge_cap = None
    if args.dgraph_edge_cap < 0:
        args.dgraph_edge_cap = None

    log(f"build FP-FraudSim profile={args.profile}, caps={caps}")
    tx = load_normalized_transactions(caps, args.seed)
    tx = assign_time_splits(tx)
    tx = finalize_transaction_columns(tx)

    log("build entity profiles")
    user_profile = build_user_profile(tx)
    merchant_profile = build_merchant_profile(tx)
    device_profile = build_device_profile(tx)
    ip_profile = build_ip_profile(tx)

    graph_nodes, graph_edges = build_graph(tx, args)
    files = write_outputs(tx, user_profile, merchant_profile, device_profile,
                          ip_profile, graph_nodes, graph_edges, args.stream_limit)
    write_manifest(tx, graph_nodes, graph_edges, files, args)

    log("done")
    log(f"  transactions={len(tx):,}, fraud={(tx['is_fraud'] == 1).sum():,}, "
        f"graph_edges={len(graph_edges):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
