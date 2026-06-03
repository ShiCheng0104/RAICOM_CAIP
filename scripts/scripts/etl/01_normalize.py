"""
ETL Stage ① 字段标准化 (FP-FraudSim)

把多源原始数据转成统一 schema 的 parquet 输出到
    data/processed/raw_unified/<source>.parquet

统一字段（最小核心列，所有源都有）:
    transaction_id : str               # 全局唯一主键 = "<source>:<原ID或行号>"
    source         : str               # paysim/banksim/ieee_cis/saml_d/creditcard
    timestamp      : datetime64[ns]    # 统一时间轴，基准 BASE_DATE
    amount         : float64
    currency       : str               # 默认 USD，SAML-D 用源币
    txn_type       : str
    payer_id       : str
    payee_id       : str               # 可能与 merchant_id 重合
    merchant_id    : str | None
    merchant_category : str | None
    payer_country  : str | None
    payee_country  : str | None
    is_fraud       : int8 (0/1)
    fraud_subtype  : str | None        # SAML-D 的 Laundering_type 等

源专属高维特征（IEEE 的 V/C/D/M/id_*、CreditCard 的 V1..V28 等）写到
    data/processed/raw_unified/<source>_features.parquet
    以 transaction_id 关联。

运行:
    D:\\Anaconda3\\python.exe scripts\\etl\\01_normalize.py
    D:\\Anaconda3\\python.exe scripts\\etl\\01_normalize.py --only paysim,saml_d
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "processed" / "raw_unified"
OUT.mkdir(parents=True, exist_ok=True)

BASE_DATE = pd.Timestamp("2026-01-01 00:00:00")
CORE_COLS = [
    "transaction_id", "source", "timestamp", "amount", "currency",
    "txn_type", "payer_id", "payee_id", "merchant_id",
    "merchant_category", "payer_country", "payee_country",
    "is_fraud", "fraud_subtype",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_parquet(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_parquet(path, index=False, compression="snappy")
    log(f"  -> {path.relative_to(ROOT)}  shape={df.shape}")


# ---------------------------------------------------------------- PaySim
def normalize_paysim() -> None:
    src = DATA / "paysim" / "PS_20174392719_1491204439457_log.csv"
    log(f"[paysim] {src.name}")
    df = pd.read_csv(src)
    n = len(df)
    tid = pd.Series(np.arange(n), dtype="int64").astype(str).radd("paysim:")
    out = pd.DataFrame({
        "transaction_id": tid,
        "source": "paysim",
        "timestamp": BASE_DATE + pd.to_timedelta(df["step"].astype(int), unit="h"),
        "amount": df["amount"].astype("float64"),
        "currency": "USD",
        "txn_type": df["type"].str.lower(),
        "payer_id": df["nameOrig"].astype(str).radd("paysim_u:"),
        "payee_id": df["nameDest"].astype(str).radd("paysim_u:"),
        "merchant_id": np.where(
            df["nameDest"].str.startswith("M"),
            df["nameDest"].astype(str).radd("paysim_m:"),
            None,
        ),
        "merchant_category": None,
        "payer_country": "SG",
        "payee_country": "SG",
        "is_fraud": df["isFraud"].astype("int8"),
        "fraud_subtype": np.where(df["isFlaggedFraud"] == 1, "rule_flagged", None),
    })
    write_parquet(out[CORE_COLS], "paysim.parquet")

    # 余额特征单独保存
    feats = pd.DataFrame({
        "transaction_id": out["transaction_id"],
        "payer_old_balance": df["oldbalanceOrg"].astype("float32"),
        "payer_new_balance": df["newbalanceOrig"].astype("float32"),
        "payee_old_balance": df["oldbalanceDest"].astype("float32"),
        "payee_new_balance": df["newbalanceDest"].astype("float32"),
    })
    write_parquet(feats, "paysim_features.parquet")


# ---------------------------------------------------------------- BankSim
def _strip_quotes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip("'")


def normalize_banksim() -> None:
    src = DATA / "banksim" / "bs140513_032310.csv"
    log(f"[banksim] {src.name}")
    df = pd.read_csv(src)
    n = len(df)
    customer = _strip_quotes(df["customer"])
    merchant = _strip_quotes(df["merchant"])
    category = _strip_quotes(df["category"])
    tid = pd.Series(np.arange(n), dtype="int64").astype(str).radd("banksim:")
    out = pd.DataFrame({
        "transaction_id": tid,
        "source": "banksim",
        "timestamp": BASE_DATE + pd.to_timedelta(df["step"].astype(int), unit="D"),
        "amount": df["amount"].astype("float64"),
        "currency": "USD",
        "txn_type": "card_payment",
        "payer_id": customer.radd("banksim_u:"),
        "payee_id": merchant.radd("banksim_m:"),
        "merchant_id": merchant.radd("banksim_m:"),
        "merchant_category": category,
        "payer_country": "ES",
        "payee_country": "ES",
        "is_fraud": df["fraud"].astype("int8"),
        "fraud_subtype": None,
    })
    write_parquet(out[CORE_COLS], "banksim.parquet")

    # 用户/商户画像特征
    user_feats = pd.DataFrame({
        "transaction_id": out["transaction_id"],
        "user_age_bucket": _strip_quotes(df["age"]),
        "user_gender": _strip_quotes(df["gender"]),
        "user_zipcode": _strip_quotes(df["zipcodeOri"]),
        "merchant_zipcode": _strip_quotes(df["zipMerchant"]),
    })
    write_parquet(user_feats, "banksim_features.parquet")


# ---------------------------------------------------------------- Credit Card Fraud
def normalize_creditcard() -> None:
    src = DATA / "creditcard" / "creditcard.csv"
    log(f"[creditcard] {src.name}")
    df = pd.read_csv(src)
    n = len(df)
    idx = pd.Series(np.arange(n), dtype="int64").astype(str)
    tid = idx.radd("cc:")
    out = pd.DataFrame({
        "transaction_id": tid,
        "source": "creditcard",
        "timestamp": BASE_DATE + pd.to_timedelta(df["Time"].astype(float), unit="s"),
        "amount": df["Amount"].astype("float64"),
        "currency": "EUR",
        "txn_type": "card_payment",
        "payer_id": idx.radd("cc_u:"),  # 匿名，无原 ID
        "payee_id": "cc_unknown",
        "merchant_id": None,
        "merchant_category": None,
        "payer_country": "EU",
        "payee_country": "EU",
        "is_fraud": df["Class"].astype("int8"),
        "fraud_subtype": None,
    })
    write_parquet(out[CORE_COLS], "creditcard.parquet")

    # PCA 特征 V1..V28
    v_cols = [f"V{i}" for i in range(1, 29)]
    feats = df[v_cols].astype("float32").copy()
    feats.insert(0, "transaction_id", tid)
    write_parquet(feats, "creditcard_features.parquet")


# ---------------------------------------------------------------- SAML-D
SAML_TYPE_MAP = {
    "Cash Deposit": "cash_in",
    "Cash Withdrawal": "cash_out",
    "Cross-border": "cross_border_transfer",
    "Make Payment": "payment",
    "Pay by Cash": "cash_payment",
    "Pay by Cheque": "cheque",
    "Online Purchase": "online_payment",
    "Local Transfer": "transfer",
    "Mobile Top-Up": "topup",
}


def normalize_saml_d() -> None:
    src = DATA / "saml_d" / "SAML-D.csv"
    log(f"[saml_d] {src.name}")
    # ~9.5M rows — 用 chunk 读
    out_parts: list[pd.DataFrame] = []
    chunks = pd.read_csv(src, chunksize=1_000_000)
    offset = 0
    for i, df in enumerate(chunks):
        df = df.reset_index(drop=True)
        ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                            errors="coerce")
        n = len(df)
        tid = pd.Series(np.arange(n) + offset,
                        dtype="int64").astype(str).radd("saml:")
        offset += n
        part = pd.DataFrame({
            "transaction_id": tid,
            "source": "saml_d",
            "timestamp": ts,
            "amount": df["Amount"].astype("float64"),
            "currency": df["Payment_currency"].astype(str),
            "txn_type": df["Payment_type"].map(SAML_TYPE_MAP).fillna(
                df["Payment_type"].astype(str).str.lower().str.replace(" ", "_")),
            "payer_id": df["Sender_account"].astype(str).radd("saml_a:"),
            "payee_id": df["Receiver_account"].astype(str).radd("saml_a:"),
            "merchant_id": None,
            "merchant_category": None,
            "payer_country": df["Sender_bank_location"].astype(str),
            "payee_country": df["Receiver_bank_location"].astype(str),
            "is_fraud": df["Is_laundering"].astype("int8"),
            "fraud_subtype": np.where(
                df["Is_laundering"] == 1,
                df["Laundering_type"].astype(str),
                None,
            ),
        })
        out_parts.append(part[CORE_COLS])
        log(f"  chunk {i}: rows={n}")
    big = pd.concat(out_parts, ignore_index=True)
    # 把统一表落盘后再考虑分片
    write_parquet(big, "saml_d.parquet")


# ---------------------------------------------------------------- IEEE-CIS
def _ieee_one_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    txn = pd.read_csv(DATA / "ieee_cis" / f"{split}_transaction.csv")
    idn_path = DATA / "ieee_cis" / f"{split}_identity.csv"
    idn = pd.read_csv(idn_path) if idn_path.exists() else pd.DataFrame(
        columns=["TransactionID"])

    n = len(txn)
    is_train = split == "train"
    tid = txn["TransactionID"].astype(str).radd("ieee:")
    out = pd.DataFrame({
        "transaction_id": tid,
        "source": "ieee_cis",
        "timestamp": BASE_DATE + pd.to_timedelta(
            txn["TransactionDT"].astype(float), unit="s"),
        "amount": txn["TransactionAmt"].astype("float64"),
        "currency": "USD",
        "txn_type": txn["ProductCD"].astype(str),
        "payer_id": txn["card1"].astype("Int64").astype(str).radd("ieee_card:"),
        "payee_id": np.where(
            txn["addr1"].notna(),
            txn["addr1"].astype("Int64").astype(str).radd("ieee_addr:"),
            "ieee_unknown",
        ),
        "merchant_id": None,
        "merchant_category": txn["ProductCD"].astype(str),
        "payer_country": txn["addr2"].astype("Int64").astype(str).where(
            txn["addr2"].notna(), None),
        "payee_country": None,
        "is_fraud": (txn["isFraud"].astype("int8") if is_train
                     else pd.Series([-1] * n, dtype="int8")),
        "fraud_subtype": None,
    })
    out["split"] = split  # 仅用于追溯，之后会被 07_split.py 重排

    # 全量特征拼上 identity，保留 transaction_id 作为主键
    feats = txn.merge(idn, on="TransactionID", how="left")
    feats.insert(0, "transaction_id", tid.values)
    feats.drop(columns=["TransactionID"], inplace=True)
    return out, feats


def normalize_ieee_cis() -> None:
    log("[ieee_cis] train_*.csv + test_*.csv")
    parts_core, parts_feat = [], []
    for split in ("train", "test"):
        log(f"  loading {split} ...")
        core, feat = _ieee_one_split(split)
        parts_core.append(core)
        parts_feat.append(feat)
    core = pd.concat(parts_core, ignore_index=True)
    write_parquet(core[CORE_COLS], "ieee_cis.parquet")
    feat = pd.concat(parts_feat, ignore_index=True)
    # 转 float32 节省磁盘
    for c in feat.columns:
        if feat[c].dtype == "float64":
            feat[c] = feat[c].astype("float32")
    write_parquet(feat, "ieee_cis_features.parquet")


# ---------------------------------------------------------------- main
SOURCES = {
    "paysim":     normalize_paysim,
    "banksim":    normalize_banksim,
    "creditcard": normalize_creditcard,
    "saml_d":     normalize_saml_d,
    "ieee_cis":   normalize_ieee_cis,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="",
                        help="逗号分隔，如 paysim,saml_d；缺省全部")
    parser.add_argument("--skip-existing", action="store_true",
                        help="若输出 parquet 已存在则跳过")
    args = parser.parse_args()

    targets = ([s.strip() for s in args.only.split(",") if s.strip()]
               or list(SOURCES))
    log(f"normalize sources: {targets}")
    failed: list[str] = []
    for name in targets:
        if name not in SOURCES:
            log(f"unknown source: {name}")
            failed.append(name)
            continue
        if args.skip_existing and (OUT / f"{name}.parquet").exists():
            log(f"[skip] {name} (parquet exists)")
            continue
        try:
            SOURCES[name]()
        except Exception as e:  # noqa: BLE001
            log(f"[fail] {name}: {type(e).__name__}: {e}")
            failed.append(name)
    log("=" * 60)
    log(f"done. failed={failed if failed else 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
