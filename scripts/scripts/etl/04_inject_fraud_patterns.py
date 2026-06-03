"""
ETL Stage ④ inject realistic fraud patterns into FP-FraudSim.

This stage keeps the clean unified dataset intact and writes an augmented copy:

    data/processed/fp_fraudsim_injected/

Injected scenarios:
    account_takeover
    phishing_transfer
    money_laundering
    mule_account
    bonus_abuse
    merchant_cashout
    device_group_fraud

Run:
    python scripts/etl/04_inject_fraud_patterns.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BASE_DIR = DATA / "processed" / "fp_fraudsim"
DEFAULT_OUT = DATA / "processed" / "fp_fraudsim_injected"
BASE_DATE = pd.Timestamp("2026-01-01 00:00:00")
END_DATE = pd.Timestamp("2026-03-31 23:59:59")
WINDOW_SECONDS = int((END_DATE - BASE_DATE).total_seconds())

DEFAULT_COUNTS = {
    "account_takeover": 3_000,
    "phishing_transfer": 3_000,
    "money_laundering": 2_400,
    "mule_account": 2_800,
    "bonus_abuse": 3_200,
    "merchant_cashout": 3_000,
    "device_group_fraud": 3_200,
}

EXTRA_TX_COLS = [
    "is_injected",
    "fraud_group_id",
    "injection_scenario",
    "scenario_step",
    "injection_evidence",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_stage2_module() -> Any:
    path = Path(__file__).with_name("02_build_unified_dataset.py")
    spec = importlib.util.spec_from_file_location("stage2_build_unified", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE2 = load_stage2_module()


def json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def choose_one(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    return df.iloc[int(rng.integers(0, len(df)))]


def choose_value(values: np.ndarray, rng: np.random.Generator) -> Any:
    return values[int(rng.integers(0, len(values)))]


def random_timestamp(rng: np.random.Generator, night_prob: float = 0.25) -> pd.Timestamp:
    seconds = int(rng.integers(0, WINDOW_SECONDS + 1))
    ts = BASE_DATE + pd.to_timedelta(seconds, unit="s")
    if rng.random() < night_prob:
        ts = ts.normalize() + pd.to_timedelta(int(rng.integers(0, 6)), unit="h")
        ts += pd.to_timedelta(int(rng.integers(0, 3600)), unit="s")
        if ts > END_DATE:
            ts = END_DATE
    return ts


def amount_from_user(user: pd.Series, rng: np.random.Generator, low: float, high: float,
                     min_amount: float, max_amount: float) -> float:
    avg = float(user.get("avg_amount", 100.0))
    if not np.isfinite(avg) or avg <= 0:
        avg = 100.0
    amount = avg * float(rng.uniform(low, high))
    amount = min(max(amount, min_amount), max_amount)
    return round(float(amount), 2)


def amount_bucket(amount: float) -> str:
    if amount < 10:
        return "micro"
    if amount < 100:
        return "small"
    if amount < 1_000:
        return "medium"
    if amount < 10_000:
        return "large"
    return "xlarge"


def split_for_timestamp(ts: pd.Timestamp, train_max: pd.Timestamp,
                        valid_max: pd.Timestamp) -> str:
    if ts <= train_max:
        return "train"
    if ts <= valid_max:
        return "valid"
    return "test"


def evidence(**kwargs: Any) -> str:
    return json.dumps(kwargs, ensure_ascii=False, default=json_default)


class FraudInjector:
    def __init__(self, base_dir: Path, out_dir: Path, seed: int) -> None:
        self.base_dir = base_dir
        self.out_dir = out_dir
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.rows: list[dict[str, Any]] = []
        self.group_rows: list[dict[str, Any]] = []
        self.edge_rows: list[dict[str, Any]] = []

        self.tx = pd.read_parquet(base_dir / "transaction_log.parquet")
        self.users = pd.read_parquet(base_dir / "user_profile.parquet")
        self.merchants = pd.read_parquet(base_dir / "merchant_profile.parquet")
        self.devices = pd.read_parquet(base_dir / "device_profile.parquet")
        self.ips = pd.read_parquet(base_dir / "ip_geo_profile.parquet")
        self.graph_nodes = pd.read_parquet(base_dir / "graph_nodes.parquet")
        self.graph_edges = pd.read_parquet(base_dir / "graph_edges.parquet")

        self.train_max = pd.read_parquet(
            base_dir / "splits" / "train.parquet", columns=["timestamp"]
        )["timestamp"].max()
        self.valid_max = pd.read_parquet(
            base_dir / "splits" / "valid.parquet", columns=["timestamp"]
        )["timestamp"].max()

        self.normal_users = self.users[
            (self.users["is_black_user"] == 0) & (self.users["txn_count"] >= 2)
        ].copy()
        if self.normal_users.empty:
            self.normal_users = self.users.copy()

        self.high_risk_merchants = self.merchants.sort_values(
            "merchant_risk_score", ascending=False
        ).head(max(1_000, min(len(self.merchants), 5_000)))
        self.high_risk_devices = self.devices.sort_values(
            "device_risk_score", ascending=False
        ).head(max(1_000, min(len(self.devices), 5_000)))
        self.high_risk_ips = self.ips.sort_values(
            "ip_risk_score", ascending=False
        ).head(max(1_000, min(len(self.ips), 5_000)))

    def next_tx_id(self, scenario: str, i: int) -> str:
        return f"inj:{scenario}:{i:07d}"

    def new_user_id(self, scenario: str, i: int) -> str:
        return f"inj_u:{scenario}:{i:07d}"

    def new_merchant_id(self, scenario: str, i: int) -> str:
        return f"inj_m:{scenario}:{i:07d}"

    def new_device_id(self, scenario: str, i: int) -> str:
        return f"D_INJ_{scenario[:3].upper()}_{i:07d}"

    def new_ip_id(self, scenario: str, i: int) -> str:
        return f"IP_INJ_{scenario[:3].upper()}_{i:07d}"

    def add_edge(self, src_id: str, dst_id: str, src_type: str, dst_type: str,
                 edge_type: str, weight: float, timestamp: pd.Timestamp,
                 label: int = 1, edge_subtype: int | None = None) -> None:
        self.edge_rows.append({
            "src_id": src_id,
            "dst_id": dst_id,
            "src_type": src_type,
            "dst_type": dst_type,
            "edge_type": edge_type,
            "weight": float(weight),
            "timestamp": timestamp,
            "source": "fraud_injection",
            "label": np.int8(label),
            "edge_subtype": edge_subtype,
        })

    def add_group_membership(self, group_id: str, user_id: str | None,
                             device_id: str | None, ip_id: str | None,
                             merchant_id: str | None, timestamp: pd.Timestamp) -> None:
        if user_id:
            self.add_edge(user_id, group_id, "user", "fraud_group",
                          "member_of_fraud_group", 1.0, timestamp)
        if device_id:
            self.add_edge(device_id, group_id, "device", "fraud_group",
                          "device_used_by_fraud_group", 1.0, timestamp)
        if ip_id:
            self.add_edge(ip_id, group_id, "ip", "fraud_group",
                          "ip_used_by_fraud_group", 1.0, timestamp)
        if merchant_id:
            self.add_edge(merchant_id, group_id, "merchant", "fraud_group",
                          "merchant_used_by_fraud_group", 1.0, timestamp)

    def add_transaction_edges(self, row: dict[str, Any]) -> None:
        tx_id = row["transaction_id"]
        ts = row["timestamp"]
        amount = row["amount"]
        self.add_edge(row["payer_id"], tx_id, "user", "transaction",
                      "make_transaction", 1.0, ts)
        self.add_edge(tx_id, row["payee_id"], "transaction", "account",
                      "transfer_to", amount, ts)
        self.add_edge(row["payer_id"], row["device_id"], "user", "device",
                      "uses_device", 1.0, ts)
        self.add_edge(row["payer_id"], row["ip_id"], "user", "ip",
                      "login_ip", 1.0, ts)
        if row.get("merchant_id"):
            self.add_edge(tx_id, row["merchant_id"], "transaction", "merchant",
                          "paid_to_merchant", amount, ts)
        if row.get("payee_id") and not row.get("merchant_id"):
            self.add_edge(row["payer_id"], row["payee_id"], "user", "user",
                          f"{row['fraud_type']}_transfer", amount, ts)
        self.add_group_membership(
            row["fraud_group_id"], row["payer_id"], row["device_id"],
            row["ip_id"], row.get("merchant_id"), ts,
        )

    def make_row(self, scenario: str, idx: int, payer_id: str, payee_id: str,
                 amount: float, timestamp: pd.Timestamp, txn_type: str,
                 channel: str, payment_method: str, device_id: str, ip_id: str,
                 group_id: str, scenario_step: str,
                 merchant_id: str | None = None, merchant_category: str = "unknown",
                 payer_country: str = "US", payee_country: str = "US",
                 rule_score: float = 0.9, evidence_payload: dict[str, Any] | None = None) -> None:
        row = {
            "row_id": -1,
            "transaction_id": self.next_tx_id(scenario, idx),
            "source": "fraud_injection",
            "timestamp": timestamp,
            "event_type": "transaction",
            "amount": round(float(amount), 2),
            "amount_bucket": amount_bucket(float(amount)),
            "currency": "USD",
            "txn_type": txn_type,
            "channel": channel,
            "payment_method": payment_method,
            "payer_id": payer_id,
            "payee_id": payee_id,
            "merchant_id": merchant_id,
            "merchant_category": merchant_category,
            "device_id": device_id,
            "ip_id": ip_id,
            "payer_country": payer_country,
            "payee_country": payee_country,
            "is_fraud": np.int8(1),
            "fraud_type": scenario,
            "risk_level": "high",
            "rule_score": round(float(rule_score), 4),
            "label_quality": "injected_fraud",
            "split": split_for_timestamp(timestamp, self.train_max, self.valid_max),
            "hour": np.int8(timestamp.hour),
            "day_of_week": np.int8(timestamp.dayofweek),
            "is_weekend": np.int8(timestamp.dayofweek in (5, 6)),
            "is_night": np.int8(0 <= timestamp.hour <= 5),
            "is_injected": np.int8(1),
            "fraud_group_id": group_id,
            "injection_scenario": scenario,
            "scenario_step": scenario_step,
            "injection_evidence": evidence(**(evidence_payload or {})),
        }
        self.rows.append(row)
        self.add_transaction_edges(row)

    def inject_account_takeover(self, count: int) -> None:
        for i in range(count):
            user = choose_one(self.normal_users, self.rng)
            payee = self.new_user_id("ato_receiver", i)
            device = self.new_device_id("account_takeover", i)
            ip = self.new_ip_id("account_takeover", i)
            amount = amount_from_user(user, self.rng, 4.0, 10.0, 800.0, 50_000.0)
            ts = random_timestamp(self.rng, night_prob=0.55)
            group = f"FG_account_takeover_{i // 5:06d}"
            self.make_row(
                "account_takeover", i, user["user_id"], payee, amount, ts,
                txn_type="transfer",
                channel=choose_value(np.array(["web", "app", "api"]), self.rng),
                payment_method="bank_transfer",
                device_id=device,
                ip_id=ip,
                group_id=group,
                scenario_step="new_device_large_transfer",
                payer_country=str(user.get("home_country", "US")),
                payee_country=choose_value(np.array(["US", "UK", "UAE", "SG"]), self.rng),
                rule_score=float(self.rng.uniform(0.88, 0.99)),
                evidence_payload={
                    "new_device": True,
                    "new_ip": True,
                    "amount_multiple_of_user_avg": round(
                        amount / max(float(user.get("avg_amount", 100.0)), 1.0), 2),
                    "night_transaction": 0 <= ts.hour <= 5,
                },
            )

    def inject_phishing_transfer(self, count: int) -> None:
        scammer_pool = [self.new_user_id("phishing_scammer", i) for i in range(max(50, count // 20))]
        for i in range(count):
            user = choose_one(self.normal_users, self.rng)
            payee = choose_value(np.array(scammer_pool), self.rng)
            device = choose_one(self.high_risk_devices, self.rng)["device_id"]
            ip = choose_one(self.high_risk_ips, self.rng)["ip_id"]
            amount = amount_from_user(user, self.rng, 2.0, 6.0, 300.0, 20_000.0)
            ts = random_timestamp(self.rng, night_prob=0.30)
            group = f"FG_phishing_transfer_{i // 20:06d}"
            self.make_row(
                "phishing_transfer", i, user["user_id"], payee, amount, ts,
                txn_type="transfer",
                channel=choose_value(np.array(["app", "web"]), self.rng),
                payment_method="wallet",
                device_id=device,
                ip_id=ip,
                group_id=group,
                scenario_step="first_payee_social_engineering",
                payer_country=str(user.get("home_country", "US")),
                payee_country=choose_value(np.array(["US", "UK", "SG"]), self.rng),
                rule_score=float(self.rng.uniform(0.78, 0.94)),
                evidence_payload={
                    "new_payee": True,
                    "remark_keyword": choose_value(np.array(["investment", "deposit", "unlock_fee", "guarantee"]), self.rng),
                    "amount_deviation": "high",
                },
            )

    def inject_money_laundering(self, count: int) -> None:
        chain_len = 5
        chains = max(1, count // chain_len)
        row_idx = 0
        cashout_merchants = [
            self.new_merchant_id("laundering_cashout", i) for i in range(max(20, chains // 5))
        ]
        for g in range(chains):
            source_user = choose_one(self.normal_users, self.rng)["user_id"]
            mules = [self.new_user_id("laundering_mule", g * 10 + j) for j in range(3)]
            hub = self.new_user_id("laundering_hub", g)
            merchant = choose_value(np.array(cashout_merchants), self.rng)
            device = self.new_device_id("money_laundering", g)
            ip = self.new_ip_id("money_laundering", g)
            start = random_timestamp(self.rng, night_prob=0.35)
            base_amount = round(float(self.rng.uniform(2_000, 30_000)), 2)
            path = [source_user, *mules, hub, merchant]
            group = f"FG_money_laundering_{g:06d}"
            for step in range(chain_len):
                amount = round(base_amount * float(self.rng.uniform(0.92, 1.03)), 2)
                ts = start + pd.to_timedelta(int(step * self.rng.integers(4, 18)), unit="m")
                is_last = step == chain_len - 1
                self.make_row(
                    "money_laundering", row_idx, path[step], path[step + 1], amount, ts,
                    txn_type="cash_out" if is_last else "transfer",
                    channel=choose_value(np.array(["bank_api", "web", "api"]), self.rng),
                    payment_method="bank_transfer",
                    device_id=device,
                    ip_id=ip,
                    group_id=group,
                    scenario_step=f"chain_step_{step + 1}",
                    merchant_id=merchant if is_last else None,
                    merchant_category="cashout_service" if is_last else "aml_transfer",
                    payer_country=choose_value(np.array(["US", "UK", "SG"]), self.rng),
                    payee_country=choose_value(np.array(["US", "UK", "UAE", "SG"]), self.rng),
                    rule_score=float(self.rng.uniform(0.90, 0.99)),
                    evidence_payload={
                        "chain_length": chain_len,
                        "amount_similarity": "high",
                        "short_retention": True,
                        "step": step + 1,
                    },
                )
                row_idx += 1

    def inject_mule_account(self, count: int) -> None:
        tx_per_mule = 8
        groups = max(1, count // tx_per_mule)
        row_idx = 0
        for g in range(groups):
            mule = self.new_user_id("mule_account", g)
            hub = self.new_user_id("mule_hub", g)
            device = self.new_device_id("mule_account", g)
            ip = self.new_ip_id("mule_account", g)
            start = random_timestamp(self.rng, night_prob=0.40)
            group = f"FG_mule_account_{g:06d}"
            inbound = tx_per_mule - 2
            for j in range(inbound):
                payer = choose_one(self.normal_users, self.rng)
                amount = amount_from_user(payer, self.rng, 1.0, 4.0, 200.0, 8_000.0)
                ts = start + pd.to_timedelta(int(j * self.rng.integers(2, 8)), unit="m")
                self.make_row(
                    "mule_account", row_idx, payer["user_id"], mule, amount, ts,
                    txn_type="transfer",
                    channel=choose_value(np.array(["app", "web"]), self.rng),
                    payment_method="wallet",
                    device_id=device,
                    ip_id=ip,
                    group_id=group,
                    scenario_step="rapid_inbound",
                    payer_country=str(payer.get("home_country", "US")),
                    payee_country="US",
                    rule_score=float(self.rng.uniform(0.82, 0.95)),
                    evidence_payload={"mule": mule, "rapid_inbound": True},
                )
                row_idx += 1
            for j in range(2):
                amount = round(float(self.rng.uniform(4_000, 20_000)), 2)
                ts = start + pd.to_timedelta(int((inbound + j) * self.rng.integers(3, 10)), unit="m")
                self.make_row(
                    "mule_account", row_idx, mule, hub, amount, ts,
                    txn_type="transfer",
                    channel="api",
                    payment_method="bank_transfer",
                    device_id=device,
                    ip_id=ip,
                    group_id=group,
                    scenario_step="rapid_outbound",
                    payer_country="US",
                    payee_country=choose_value(np.array(["US", "UAE", "SG"]), self.rng),
                    rule_score=float(self.rng.uniform(0.88, 0.98)),
                    evidence_payload={"mule": mule, "rapid_outbound": True, "short_balance_retention": True},
                )
                row_idx += 1

    def inject_bonus_abuse(self, count: int) -> None:
        group_size = 10
        groups = max(1, count // group_size)
        row_idx = 0
        merchants = [self.new_merchant_id("bonus_campaign", i) for i in range(max(20, groups // 4))]
        for g in range(groups):
            device = self.new_device_id("bonus_abuse", g)
            ip = self.new_ip_id("bonus_abuse", g)
            merchant = choose_value(np.array(merchants), self.rng)
            activity = f"ACT_bonus_{g % 30:04d}"
            start = random_timestamp(self.rng, night_prob=0.10)
            group = f"FG_bonus_abuse_{g:06d}"
            for j in range(group_size):
                user = self.new_user_id("bonus_abuse", g * group_size + j)
                amount = round(float(self.rng.uniform(1.0, 60.0)), 2)
                ts = start + pd.to_timedelta(int(j * self.rng.integers(10, 90)), unit="s")
                self.make_row(
                    "bonus_abuse", row_idx, user, merchant, amount, ts,
                    txn_type="payment",
                    channel=choose_value(np.array(["mini_program", "app", "qr_code"]), self.rng),
                    payment_method="wallet",
                    device_id=device,
                    ip_id=ip,
                    group_id=group,
                    scenario_step="same_device_campaign_payment",
                    merchant_id=merchant,
                    merchant_category="campaign_subsidy",
                    payer_country="US",
                    payee_country="US",
                    rule_score=float(self.rng.uniform(0.74, 0.90)),
                    evidence_payload={
                        "activity_id": activity,
                        "same_device": True,
                        "same_ip": True,
                        "small_amount": True,
                    },
                )
                row_idx += 1

    def inject_merchant_cashout(self, count: int) -> None:
        merchants = [self.new_merchant_id("merchant_cashout", i) for i in range(max(20, count // 120))]
        row_idx = 0
        while row_idx < count:
            merchant = choose_value(np.array(merchants), self.rng)
            start = random_timestamp(self.rng, night_prob=0.25)
            group = f"FG_merchant_cashout_{row_idx // 120:06d}"
            burst = min(int(self.rng.integers(60, 130)), count - row_idx)
            ip = self.new_ip_id("merchant_cashout", row_idx // 120)
            for j in range(burst):
                user = choose_one(self.normal_users, self.rng)
                amount = float(choose_value(np.array([500, 1000, 2000, 3000, 5000, 8000]), self.rng))
                ts = start + pd.to_timedelta(int(j * self.rng.integers(5, 40)), unit="s")
                device = choose_one(self.high_risk_devices, self.rng)["device_id"]
                self.make_row(
                    "merchant_cashout", row_idx, user["user_id"], merchant, amount, ts,
                    txn_type="payment",
                    channel=choose_value(np.array(["pos", "qr_code", "app"]), self.rng),
                    payment_method="card",
                    device_id=device,
                    ip_id=ip,
                    group_id=group,
                    scenario_step="integer_amount_burst_to_merchant",
                    merchant_id=merchant,
                    merchant_category="high_risk_cashout",
                    payer_country=str(user.get("home_country", "US")),
                    payee_country="US",
                    rule_score=float(self.rng.uniform(0.82, 0.96)),
                    evidence_payload={
                        "integer_amount": True,
                        "merchant_burst": True,
                        "shared_cashout_merchant": merchant,
                    },
                )
                row_idx += 1

    def inject_device_group_fraud(self, count: int) -> None:
        group_size = 16
        groups = max(1, count // group_size)
        merchants = [self.new_merchant_id("device_group", i) for i in range(max(20, groups // 3))]
        row_idx = 0
        for g in range(groups):
            devices = [self.new_device_id("device_group_fraud", g * 3 + j) for j in range(3)]
            ips = [self.new_ip_id("device_group_fraud", g * 2 + j) for j in range(2)]
            merchant = choose_value(np.array(merchants), self.rng)
            start = random_timestamp(self.rng, night_prob=0.35)
            group = f"FG_device_group_fraud_{g:06d}"
            for j in range(group_size):
                user = self.new_user_id("device_group_fraud", g * group_size + j)
                device = choose_value(np.array(devices), self.rng)
                ip = choose_value(np.array(ips), self.rng)
                amount = round(float(self.rng.uniform(200, 6_000)), 2)
                ts = start + pd.to_timedelta(int(j * self.rng.integers(20, 180)), unit="s")
                self.make_row(
                    "device_group_fraud", row_idx, user, merchant, amount, ts,
                    txn_type=choose_value(np.array(["payment", "transfer"]), self.rng),
                    channel=choose_value(np.array(["api", "web", "app"]), self.rng),
                    payment_method="wallet",
                    device_id=device,
                    ip_id=ip,
                    group_id=group,
                    scenario_step="shared_device_ip_cluster",
                    merchant_id=merchant,
                    merchant_category="fraud_cluster_merchant",
                    payer_country="US",
                    payee_country="US",
                    rule_score=float(self.rng.uniform(0.86, 0.99)),
                    evidence_payload={
                        "shared_devices": devices,
                        "shared_ips": ips,
                        "dense_subgraph": True,
                    },
                )
                row_idx += 1

    def run(self, counts: dict[str, int], stream_limit: int) -> None:
        self.inject_account_takeover(counts["account_takeover"])
        self.inject_phishing_transfer(counts["phishing_transfer"])
        self.inject_money_laundering(counts["money_laundering"])
        self.inject_mule_account(counts["mule_account"])
        self.inject_bonus_abuse(counts["bonus_abuse"])
        self.inject_merchant_cashout(counts["merchant_cashout"])
        self.inject_device_group_fraud(counts["device_group_fraud"])
        self.write_outputs(counts, stream_limit)

    def write_outputs(self, requested_counts: dict[str, int], stream_limit: int) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "splits").mkdir(parents=True, exist_ok=True)

        base_tx = self.tx.copy()
        for col in EXTRA_TX_COLS:
            if col not in base_tx.columns:
                base_tx[col] = None
        base_tx["is_injected"] = np.int8(0)

        injected = pd.DataFrame(self.rows)
        augmented = pd.concat([base_tx, injected], ignore_index=True, sort=False)
        augmented = augmented.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
        augmented["row_id"] = np.arange(len(augmented), dtype=np.int64)

        log("rebuild profiles with injected transactions")
        user_profile = STAGE2.build_user_profile(augmented)
        merchant_profile = STAGE2.build_merchant_profile(augmented)
        device_profile = STAGE2.build_device_profile(augmented)
        ip_profile = STAGE2.build_ip_profile(augmented)

        new_edges = pd.DataFrame(self.edge_rows)
        new_edges["edge_subtype"] = new_edges["edge_subtype"].astype("Int16")
        graph_edges = pd.concat([self.graph_edges, new_edges], ignore_index=True, sort=False)
        endpoint_nodes = STAGE2.build_endpoint_nodes(new_edges)
        explicit_nodes = self.explicit_injected_nodes(injected)
        graph_nodes = pd.concat(
            [self.graph_nodes, explicit_nodes, endpoint_nodes],
            ignore_index=True,
            sort=False,
        ).drop_duplicates("node_id").reset_index(drop=True)

        log("write injected dataset")
        augmented.to_parquet(self.out_dir / "transaction_log.parquet", index=False, compression="snappy")
        user_profile.to_parquet(self.out_dir / "user_profile.parquet", index=False, compression="snappy")
        merchant_profile.to_parquet(self.out_dir / "merchant_profile.parquet", index=False, compression="snappy")
        device_profile.to_parquet(self.out_dir / "device_profile.parquet", index=False, compression="snappy")
        ip_profile.to_parquet(self.out_dir / "ip_geo_profile.parquet", index=False, compression="snappy")
        graph_nodes.to_parquet(self.out_dir / "graph_nodes.parquet", index=False, compression="snappy")
        graph_edges.to_parquet(self.out_dir / "graph_edges.parquet", index=False, compression="snappy")

        for split in ("train", "valid", "test", "unlabeled"):
            part = augmented[augmented["split"] == split]
            part.to_parquet(self.out_dir / "splits" / f"{split}.parquet",
                            index=False, compression="snappy")

        self.write_stream(augmented, injected, stream_limit)
        self.copy_feature_store()
        self.write_manifest(augmented, graph_nodes, graph_edges, requested_counts, injected)
        self.write_readme(augmented, injected)

    def explicit_injected_nodes(self, injected: pd.DataFrame) -> pd.DataFrame:
        node_rows: list[dict[str, Any]] = []
        for _, row in injected.iterrows():
            node_rows.append({
                "node_id": row["transaction_id"],
                "node_type": "transaction",
                "source": "fraud_injection",
                "label": 1,
                "raw_label": row["fraud_type"],
                "time_step": pd.NA,
                "timestamp": row["timestamp"],
            })
            node_rows.append({
                "node_id": row["fraud_group_id"],
                "node_type": "fraud_group",
                "source": "fraud_injection",
                "label": 1,
                "raw_label": row["fraud_type"],
                "time_step": pd.NA,
                "timestamp": row["timestamp"],
            })
            entity_columns: list[tuple[str, str]] = [("payer_id", "user")]
            if row.get("payee_id") and row.get("payee_id") != row.get("merchant_id"):
                entity_columns.append(("payee_id", "user"))
            entity_columns.extend([
                ("merchant_id", "merchant"),
                ("device_id", "device"),
                ("ip_id", "ip"),
            ])
            for col, node_type in entity_columns:
                value = row.get(col)
                if value:
                    node_rows.append({
                        "node_id": value,
                        "node_type": node_type,
                        "source": "fraud_injection",
                        "label": 1,
                        "raw_label": row["fraud_type"],
                        "time_step": pd.NA,
                        "timestamp": row["timestamp"],
                    })
        return pd.DataFrame(node_rows)

    def write_stream(self, augmented: pd.DataFrame, injected: pd.DataFrame, stream_limit: int) -> None:
        stream_cols = [
            "timestamp", "transaction_id", "source", "payer_id", "payee_id",
            "merchant_id", "amount", "currency", "txn_type", "channel",
            "payment_method", "device_id", "ip_id", "is_fraud", "fraud_type",
            "risk_level", "rule_score", "is_injected", "fraud_group_id",
            "injection_scenario", "scenario_step", "injection_evidence",
        ]
        injected_ids = set(injected["transaction_id"])
        injected_part = augmented[augmented["transaction_id"].isin(injected_ids)]
        if len(injected_part) >= stream_limit:
            stream = injected_part.sort_values("timestamp").head(stream_limit)
        else:
            normal_needed = stream_limit - len(injected_part)
            normal_part = augmented[~augmented["transaction_id"].isin(injected_ids)]
            normal_part = normal_part.sort_values("timestamp").head(normal_needed)
            stream = pd.concat([normal_part, injected_part], ignore_index=True).sort_values("timestamp")
        stream[stream_cols].to_json(
            self.out_dir / "transaction_stream.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
            date_format="iso",
        )

    def copy_feature_store(self) -> None:
        src = self.base_dir / "dgraphfin_node_features_sample.parquet"
        if src.exists():
            shutil.copy2(src, self.out_dir / src.name)

        index = pd.read_parquet(self.base_dir / "feature_store_index.parquet")
        index = index.copy()
        index.loc[index["table"] == "dgraphfin_node_features_sample", "path"] = (
            "data/processed/fp_fraudsim_injected/dgraphfin_node_features_sample.parquet"
        )
        index.to_parquet(self.out_dir / "feature_store_index.parquet",
                         index=False, compression="snappy")

    def write_manifest(self, augmented: pd.DataFrame, graph_nodes: pd.DataFrame,
                       graph_edges: pd.DataFrame, requested_counts: dict[str, int],
                       injected: pd.DataFrame) -> None:
        injected_counts = injected["fraud_type"].value_counts().to_dict()
        split_counts = augmented["split"].value_counts().to_dict()
        source_counts = augmented.groupby("source").agg(
            rows=("transaction_id", "size"),
            fraud=("is_fraud", lambda s: int((s == 1).sum())),
            unlabeled=("is_fraud", lambda s: int((s < 0).sum())),
        ).reset_index()
        manifest = {
            "dataset": "FP-FraudSim-Injected",
            "created_at": pd.Timestamp.now().isoformat(),
            "base_dataset": str(self.base_dir.relative_to(ROOT)),
            "seed": self.seed,
            "description": "FP-FraudSim with realistic injected fraud patterns.",
            "injection": {
                "requested_counts": requested_counts,
                "actual_counts": {str(k): int(v) for k, v in injected_counts.items()},
                "total_injected_transactions": int(len(injected)),
                "scenarios": sorted(injected_counts),
            },
            "time_window": {
                "start": str(augmented["timestamp"].min()),
                "end": str(augmented["timestamp"].max()),
                "split_policy": "Base splits are preserved; injected rows are assigned by timestamp using base train/valid/test boundaries.",
            },
            "row_counts": {
                "transaction_log": int(len(augmented)),
                "user_profile": int(len(pd.read_parquet(self.out_dir / "user_profile.parquet", columns=["user_id"]))),
                "merchant_profile": int(len(pd.read_parquet(self.out_dir / "merchant_profile.parquet", columns=["merchant_id"]))),
                "device_profile": int(len(pd.read_parquet(self.out_dir / "device_profile.parquet", columns=["device_id"]))),
                "ip_geo_profile": int(len(pd.read_parquet(self.out_dir / "ip_geo_profile.parquet", columns=["ip_id"]))),
                "graph_nodes": int(len(graph_nodes)),
                "graph_edges": int(len(graph_edges)),
            },
            "splits": {str(k): int(v) for k, v in split_counts.items()},
            "fraud": {
                "fraud_rows": int((augmented["is_fraud"] == 1).sum()),
                "normal_rows": int((augmented["is_fraud"] == 0).sum()),
                "unlabeled_rows": int((augmented["is_fraud"] < 0).sum()),
                "fraud_rate_labeled": float(
                    (augmented["is_fraud"] == 1).sum()
                    / max(int(augmented["is_fraud"].isin([0, 1]).sum()), 1)
                ),
            },
            "sources": source_counts.to_dict(orient="records"),
        }
        write_json(self.out_dir / "manifest.json", manifest)
        write_json(self.out_dir / "injection_report.json", manifest["injection"])

    def write_readme(self, augmented: pd.DataFrame, injected: pd.DataFrame) -> None:
        counts = {str(k): int(v) for k, v in injected["fraud_type"].value_counts().to_dict().items()}
        split_counts = {str(k): int(v) for k, v in augmented["split"].value_counts().to_dict().items()}
        counts_json = json.dumps(counts, ensure_ascii=False, indent=2)
        split_json = json.dumps(split_counts, ensure_ascii=False, indent=2)
        readme = f"""# FP-FraudSim-Injected

本目录是 `data/processed/fp_fraudsim` 的欺诈模式增强版，用于方向一后续的离线训练、图特征构建、Kafka/Flink 流式回放和实时风控演示。

## 1. 数据规模

- 交易总数：{len(augmented):,}
- 注入欺诈交易：{len(injected):,}
- 欺诈样本：{(augmented["is_fraud"] == 1).sum():,}
- 正常样本：{(augmented["is_fraud"] == 0).sum():,}
- 未标注样本：{(augmented["is_fraud"] < 0).sum():,}

## 2. 数据切分

{split_json}

## 3. 注入欺诈场景

{counts_json}

| fraud_type | 构造逻辑 | 主要风险信号 |
| --- | --- | --- |
| `account_takeover` | 老用户突然在新设备、新 IP、夜间或非常用渠道发起高额交易 | 金额偏离、异地、新设备、夜间交易 |
| `phishing_transfer` | 用户向首次出现的陌生收款方转账，金额偏高且渠道偏向 app/web | 新收款方、大额、社工/钓鱼转账 |
| `money_laundering` | 多跳资金链路，资金经多个中间账户快速流向高风险商户 | 多跳转账、短时间间隔、金额相近、链路团伙 |
| `mule_account` | 跑分/中转账户短时间大量收款后快速转出 | 入出度异常、资金停留时间短、新账户 |
| `bonus_abuse` | 多个账号共用少量设备和 IP，集中进行小额薅羊毛交易 | 共享设备、共享 IP、小额高频、团伙聚集 |
| `merchant_cashout` | 高风险商户短时间集中收款，大量整数金额并快速累积 | 商户交易暴涨、整数金额、集中收款 |
| `device_group_fraud` | 一批账号共用设备/IP，向少数商户或账户集中交易 | 设备团伙、IP 团伙、密集子图 |

## 4. 新增字段

- `is_injected`：是否为本脚本注入的欺诈交易，注入样本为 1，原始统一数据集样本为 0。
- `fraud_group_id`：欺诈团伙、案件或链路编号，可用于图谱溯源和按团伙聚合评估。
- `injection_scenario`：注入场景名，与 `fraud_type` 保持一致。
- `scenario_step`：交易在欺诈场景中的角色，例如 `takeover_payment`、`laundering_hop_2`、`cashout_payment`。
- `injection_evidence`：JSON 字符串，记录可解释证据，例如新设备、共享 IP、链路位置、异常倍数等。

## 5. 文件说明

- `transaction_log.parquet`
- `user_profile.parquet`
- `merchant_profile.parquet`
- `device_profile.parquet`
- `ip_geo_profile.parquet`
- `graph_nodes.parquet`
- `graph_edges.parquet`
- `transaction_stream.jsonl`
- `splits/train.parquet`
- `splits/valid.parquet`
- `splits/test.parquet`
- `splits/unlabeled.parquet`
- `manifest.json`
- `injection_report.json`

`transaction_stream.jsonl` 按时间排序，可直接作为 Kafka Producer 的回放输入；`graph_edges.parquet` 已追加欺诈团伙、设备共享、IP 共享、洗钱链路等边，适合后续做 NetworkX / Node2Vec / GNN 特征。

## 6. 重新生成与校验

重新生成：

```powershell
python scripts\\etl\\04_inject_fraud_patterns.py
```

校验：

```powershell
python scripts\\etl\\03_validate_unified_dataset.py --dataset-dir data\\processed\\fp_fraudsim_injected
```
"""
        (self.out_dir / "README.md").write_text(readme, encoding="utf-8-sig")


def parse_counts(raw_counts: list[str]) -> dict[str, int]:
    counts = DEFAULT_COUNTS.copy()
    for item in raw_counts:
        if "=" not in item:
            raise ValueError(f"Invalid --count value: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in counts:
            raise ValueError(f"Unknown scenario: {name}")
        counts[name] = int(value)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--stream-limit", type=int, default=120_000)
    parser.add_argument("--count", action="append", default=[],
                        help="Override scenario count, e.g. --count account_takeover=5000")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    counts = parse_counts(args.count)
    log(f"inject fraud patterns from {input_dir.relative_to(ROOT)} -> {output_dir.relative_to(ROOT)}")
    log(f"counts={counts}")

    injector = FraudInjector(input_dir, output_dir, args.seed)
    injector.run(counts, args.stream_limit)
    log(f"done. injected={len(injector.rows):,}, output={output_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
