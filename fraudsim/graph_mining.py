from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

from fraudsim.config import dataset_dir, load_config


MINING_DIR_NAME = "graph_mining"
ENTITY_FEATURE_COLUMNS = [
    "fraud_group_mining_id",
    "graph_mining_group_risk_score",
    "graph_mining_group_entity_count",
    "graph_mining_group_user_count",
    "graph_mining_group_resource_count",
    "graph_mining_fraud_seed_count",
    "graph_mining_fraud_seed_ratio",
    "graph_mining_shared_device_count",
    "graph_mining_shared_ip_count",
    "graph_mining_shared_merchant_count",
    "graph_mining_shared_payee_count",
    "graph_mining_evidence_count",
    "graph_mining_scenario_count",
]


@dataclass
class UnionFind:
    parent: dict[str, str]
    size: dict[str, int]

    def __init__(self) -> None:
        self.parent = {}
        self.size = {}

    def add(self, item: str | None) -> None:
        if not item:
            return
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str | None, right: str | None) -> None:
        if not left or not right:
            return
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def mining_dir(ds_dir: Path) -> Path:
    return ds_dir / MINING_DIR_NAME


def entity_risk_path(ds_dir: Path) -> Path:
    return mining_dir(ds_dir) / "entity_graph_risk.parquet"


def groups_path(ds_dir: Path) -> Path:
    return mining_dir(ds_dir) / "fraud_groups.parquet"


def evidence_path(ds_dir: Path) -> Path:
    return mining_dir(ds_dir) / "group_evidence.jsonl"


def summary_path(ds_dir: Path) -> Path:
    return mining_dir(ds_dir) / "graph_mining_summary.json"


def _read_splits(ds_dir: Path, splits: Iterable[str]) -> pd.DataFrame:
    frames = []
    base_cols = [
        "transaction_id",
        "source",
        "timestamp",
        "amount",
        "payer_id",
        "payee_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "is_fraud",
        "is_injected",
        "fraud_group_id",
        "injection_scenario",
        "scenario_step",
    ]
    for split in splits:
        path = ds_dir / "splits" / f"{split}.parquet"
        if not path.exists():
            continue
        cols = pq.read_schema(path).names
        use_cols = [col for col in base_cols if col in cols]
        frame = pd.read_parquet(path, columns=use_cols)
        frame["_fraudsim_split"] = split
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=base_cols + ["_fraudsim_split"])
    out = pd.concat(frames, ignore_index=True)
    for col in base_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["is_fraud"] = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(0).astype("int8")
    out["is_injected"] = pd.to_numeric(out["is_injected"], errors="coerce").fillna(0).astype("int8")
    return out


def _resource_stats(
    tx: pd.DataFrame,
    resource_col: str,
    resource_type: str,
    min_users: int,
    max_users: int,
) -> pd.DataFrame:
    rows = tx.dropna(subset=["payer_id", resource_col]).copy()
    if rows.empty:
        return pd.DataFrame()
    rows[resource_col] = rows[resource_col].astype(str)
    rows["payer_id"] = rows["payer_id"].astype(str)
    stats = rows.groupby(resource_col, as_index=False).agg(
        entity_id=(resource_col, "first"),
        entity_type=(resource_col, lambda _: resource_type),
        shared_user_count=("payer_id", "nunique"),
        txn_count=("transaction_id", "count"),
        amount_sum=("amount", "sum"),
        fraud_seed_count=("is_fraud", "sum"),
        injected_count=("is_injected", "sum"),
        scenario_count=("injection_scenario", lambda x: int(x.dropna().nunique())),
    )
    stats["fraud_seed_ratio"] = stats["fraud_seed_count"] / stats["txn_count"].clip(lower=1)
    stats["injected_ratio"] = stats["injected_count"] / stats["txn_count"].clip(lower=1)
    stats["evidence_kind"] = f"shared_{resource_type}"
    stats = stats[(stats["shared_user_count"] >= min_users) & (stats["shared_user_count"] <= max_users)].copy()
    if stats.empty:
        return stats
    amount_q90 = float(stats["amount_sum"].quantile(0.90))
    user_component = stats["shared_user_count"].map(lambda v: min(1.0, math.log1p(float(v)) / math.log(40)))
    volume_component = (stats["amount_sum"] / max(amount_q90, 1.0)).clip(upper=1.0)
    stats["resource_risk_score"] = (
        0.38 * stats["fraud_seed_ratio"].clip(upper=1.0)
        + 0.26 * user_component
        + 0.18 * volume_component
        + 0.10 * (stats["txn_count"] / stats["txn_count"].quantile(0.90)).clip(upper=1.0)
        + 0.08 * (stats["scenario_count"] / 3).clip(upper=1.0)
    ).clip(upper=1.0)
    return stats


def _candidate_resources(tx: pd.DataFrame, max_users_per_resource: int) -> pd.DataFrame:
    specs = [
        ("device_id", "device", 2, max_users_per_resource),
        ("ip_id", "ip", 3, max_users_per_resource),
        ("merchant_id", "merchant", 4, max_users_per_resource),
        ("payee_id", "payee", 3, max_users_per_resource),
    ]
    frames = []
    for resource_col, resource_type, min_users, max_users in specs:
        if resource_col not in tx.columns:
            continue
        stats = _resource_stats(tx, resource_col, resource_type, min_users, max_users)
        if stats.empty:
            continue
        min_fraud = 1 if resource_type in {"device", "ip"} else 2
        selected = stats[
            (stats["fraud_seed_count"] >= min_fraud)
            | ((stats["resource_risk_score"] >= 0.45) & (stats["txn_count"] >= min_users * 2))
        ].copy()
        selected["resource_col"] = resource_col
        frames.append(selected)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _typed_counts(members: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for member in members:
        if member.startswith("D_"):
            counts["device"] += 1
        elif member.startswith("IP_"):
            counts["ip"] += 1
        elif member.startswith("banksim_m:") or member.startswith("amlsim_branch:") or member.startswith("M_"):
            counts["merchant"] += 1
        elif member.startswith("payee:"):
            counts["payee"] += 1
        else:
            counts["user"] += 1
    return dict(counts)


def _component_risk(
    user_count: int,
    entity_count: int,
    fraud_seed_count: int,
    txn_count: int,
    shared_device_count: int,
    shared_ip_count: int,
    evidence_count: int,
    scenario_count: int,
) -> float:
    fraud_ratio = fraud_seed_count / max(txn_count, 1)
    user_component = min(1.0, math.log1p(user_count) / math.log(30))
    evidence_component = min(1.0, math.log1p(evidence_count) / math.log(20))
    resource_component = min(1.0, (shared_device_count + shared_ip_count) / max(user_count, 1))
    scenario_component = min(1.0, scenario_count / 3)
    density_component = min(1.0, entity_count / max(user_count * 2, 1))
    return float(min(1.0, (
        0.40 * fraud_ratio
        + 0.20 * user_component
        + 0.17 * evidence_component
        + 0.13 * resource_component
        + 0.06 * scenario_component
        + 0.04 * density_component
    )))


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.65:
        return "medium"
    return "watch"


def group_explanations(group: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    codes: list[str] = []
    evidence_counter = Counter(str(item.get("evidence_kind") or "unknown") for item in evidence)
    scenario_map = json.loads(group.get("top_scenarios") or "{}") if isinstance(group.get("top_scenarios"), str) else group.get("top_scenarios") or {}

    if float(group.get("graph_mining_group_risk_score") or 0.0) >= 0.75:
        codes.append("high_confidence_ring")
    if int(group.get("shared_device_count") or 0) >= 2 or evidence_counter.get("shared_device", 0) >= 2:
        codes.append("shared_device_ring")
    if int(group.get("shared_ip_count") or 0) >= 2 or evidence_counter.get("shared_ip", 0) >= 2:
        codes.append("shared_ip_cluster")
    if int(group.get("shared_merchant_count") or 0) >= 1 or evidence_counter.get("shared_merchant", 0) >= 1:
        codes.append("merchant_cashout_cluster")
    if int(group.get("shared_payee_count") or 0) >= 1 or evidence_counter.get("shared_payee", 0) >= 1:
        codes.append("shared_payee_chain")
    if int(group.get("fraud_seed_count") or 0) >= 3:
        codes.append("fraud_seed_propagation")
    if int(group.get("scenario_count") or 0) >= 2:
        codes.append("multi_scenario_campaign")
    if any("money_laundering" in str(key) or "mule" in str(key) for key in scenario_map):
        codes.append("fund_transfer_chain")

    codes = list(dict.fromkeys(codes))
    if not codes:
        codes.append("weak_graph_signal")

    evidence_summary = {
        "by_kind": dict(evidence_counter),
        "top_resources": [
            {
                "entity_id": item.get("entity_id"),
                "entity_type": item.get("entity_type"),
                "evidence_kind": item.get("evidence_kind"),
                "shared_user_count": int(item.get("shared_user_count", 0)),
                "fraud_seed_count": int(item.get("fraud_seed_count", 0)),
                "resource_risk_score": float(item.get("resource_risk_score", 0.0)),
            }
            for item in sorted(evidence, key=lambda row: float(row.get("resource_risk_score", 0.0)), reverse=True)[:5]
        ],
    }
    explanation_text = (
        f"团伙风险分 {float(group.get('graph_mining_group_risk_score') or 0.0):.3f}，"
        f"覆盖 {int(group.get('user_count') or 0)} 个用户、{int(group.get('resource_count') or 0)} 个共享资源，"
        f"包含 {int(group.get('fraud_seed_count') or 0)} 条历史欺诈种子；"
        f"主要证据为 {', '.join(codes[:4])}。"
    )
    return {
        "risk_level": risk_level(float(group.get("graph_mining_group_risk_score") or 0.0)),
        "explanation_codes": codes,
        "explanation_text": explanation_text,
        "evidence_summary": evidence_summary,
    }


def build_group_subgraph(group: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    group_id = str(group.get("fraud_group_mining_id") or "group")
    users = json.loads(group.get("sample_users") or "[]") if isinstance(group.get("sample_users"), str) else group.get("sample_users") or []
    resources = json.loads(group.get("sample_resources") or "[]") if isinstance(group.get("sample_resources"), str) else group.get("sample_resources") or []
    evidence_by_id = {str(item.get("entity_id")): item for item in evidence}
    nodes = [
        {
            "id": group_id,
            "label": group_id,
            "type": "fraud_group",
            "role": "center",
            "risk_score": float(group.get("graph_mining_group_risk_score") or 0.0),
        },
        {
            "id": f"{group_id}:risk",
            "label": risk_level(float(group.get("graph_mining_group_risk_score") or 0.0)),
            "type": "risk",
            "role": "risk",
            "risk_score": float(group.get("graph_mining_group_risk_score") or 0.0),
        },
    ]
    edges = [{"source": group_id, "target": f"{group_id}:risk", "label": "risk_score"}]
    for user_id in users[:10]:
        nodes.append({"id": str(user_id), "label": str(user_id), "type": "user", "role": "member"})
        edges.append({"source": str(user_id), "target": group_id, "label": "member"})
    for resource_id in resources[:12]:
        evidence_row = evidence_by_id.get(str(resource_id), {})
        entity_type = evidence_row.get("entity_type")
        if not entity_type:
            entity_type = "ip" if str(resource_id).startswith("IP_") else "device" if str(resource_id).startswith("D_") else "resource"
        nodes.append({
            "id": str(resource_id),
            "label": str(resource_id),
            "type": entity_type,
            "role": "shared_resource",
            "risk_score": float(evidence_row.get("resource_risk_score", 0.0)),
            "shared_user_count": int(evidence_row.get("shared_user_count", 0)),
        })
        edges.append({
            "source": group_id,
            "target": str(resource_id),
            "label": evidence_row.get("evidence_kind") or "shared_resource",
            "weight": float(evidence_row.get("resource_risk_score", 0.0)),
        })
    return {"nodes": nodes, "edges": edges}


def mine_fraud_graph(
    ds_dir: Path,
    force: bool = False,
    mining_splits: tuple[str, ...] = ("train",),
    evaluation_splits: tuple[str, ...] = ("train", "valid", "test"),
    max_users_per_resource: int = 200,
    min_group_risk_score: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out_entity = entity_risk_path(ds_dir)
    out_groups = groups_path(ds_dir)
    out_summary = summary_path(ds_dir)
    if out_entity.exists() and out_groups.exists() and out_summary.exists() and not force:
        with out_summary.open("r", encoding="utf-8") as f:
            return pd.read_parquet(out_entity), pd.read_parquet(out_groups), json.load(f)

    tx_mining = _read_splits(ds_dir, mining_splits)
    tx_eval = _read_splits(ds_dir, evaluation_splits)
    resources = _candidate_resources(tx_mining, max_users_per_resource=max_users_per_resource)
    uf = UnionFind()
    evidence_by_resource: dict[str, dict[str, Any]] = {}
    resource_col_by_id: dict[str, str] = {}

    if not resources.empty:
        for row in resources.to_dict(orient="records"):
            resource_id = str(row["entity_id"])
            resource_col_by_id[resource_id] = str(row["resource_col"])
            evidence_by_resource[resource_id] = row
            uf.add(resource_id)

        for resource_col in sorted(resources["resource_col"].unique()):
            resource_ids = set(resources.loc[resources["resource_col"] == resource_col, "entity_id"].astype(str))
            rows = tx_mining[tx_mining[resource_col].astype(str).isin(resource_ids)].dropna(subset=["payer_id", resource_col])
            for row in rows[["payer_id", resource_col]].itertuples(index=False):
                payer_id = str(row[0])
                resource_id = str(row[1])
                uf.union(payer_id, resource_id)

    payer_set = set(tx_mining["payer_id"].dropna().astype(str))
    high_amount = float(tx_mining["amount"].quantile(0.95)) if not tx_mining.empty else 0.0
    chain_rows = tx_mining[
        tx_mining["payer_id"].notna()
        & tx_mining["payee_id"].notna()
        & tx_mining["payee_id"].astype(str).isin(payer_set)
        & ((tx_mining["is_fraud"] == 1) | (tx_mining["amount"] >= high_amount))
    ]
    for row in chain_rows[["payer_id", "payee_id"]].itertuples(index=False):
        uf.union(str(row[0]), str(row[1]))

    components: dict[str, set[str]] = defaultdict(set)
    for node in list(uf.parent):
        components[uf.find(node)].add(node)

    tx_by_payer = tx_mining.copy()
    tx_by_payer["payer_id"] = tx_by_payer["payer_id"].astype(str)
    component_for_user = {
        member: uf.find(member)
        for member in uf.parent
        if member in payer_set
    }
    tx_by_payer["_component"] = tx_by_payer["payer_id"].map(component_for_user)

    resource_component = {
        resource_id: uf.find(resource_id)
        for resource_id in evidence_by_resource
        if resource_id in uf.parent
    }
    evidence_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for resource_id, component in resource_component.items():
        evidence_by_component[component].append(evidence_by_resource[resource_id])

    group_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    rank = 0
    for root, members in components.items():
        users = sorted([member for member in members if member in payer_set])
        if len(users) < 2:
            continue
        component_tx = tx_by_payer[tx_by_payer["_component"] == root]
        if component_tx.empty:
            continue
        evidence = evidence_by_component.get(root, [])
        type_counts = _typed_counts(members)
        scenario_count = int(component_tx["injection_scenario"].dropna().nunique())
        risk_score = _component_risk(
            user_count=len(users),
            entity_count=len(members),
            fraud_seed_count=int(component_tx["is_fraud"].sum()),
            txn_count=len(component_tx),
            shared_device_count=type_counts.get("device", 0),
            shared_ip_count=type_counts.get("ip", 0),
            evidence_count=len(evidence),
            scenario_count=scenario_count,
        )
        if risk_score < min_group_risk_score and not evidence:
            continue
        rank += 1
        group_id = f"GM_{rank:06d}"
        fraud_groups = component_tx["fraud_group_id"].dropna().astype(str)
        scenario_counter = Counter(component_tx["injection_scenario"].dropna().astype(str))
        dominant_injected_group = fraud_groups.value_counts().index[0] if not fraud_groups.empty else None
        dominant_injected_share = (
            float(fraud_groups.value_counts().iloc[0] / max(len(fraud_groups), 1)) if not fraud_groups.empty else 0.0
        )
        group = {
            "fraud_group_mining_id": group_id,
            "graph_mining_group_risk_score": risk_score,
            "entity_count": int(len(members)),
            "user_count": int(len(users)),
            "resource_count": int(len(members) - len(users)),
            "txn_count": int(len(component_tx)),
            "amount_sum": float(component_tx["amount"].sum()),
            "fraud_seed_count": int(component_tx["is_fraud"].sum()),
            "fraud_seed_ratio": float(component_tx["is_fraud"].sum() / max(len(component_tx), 1)),
            "injected_txn_count": int(component_tx["is_injected"].sum()),
            "injected_group_count": int(fraud_groups.nunique()),
            "dominant_injected_group": dominant_injected_group,
            "dominant_injected_share": dominant_injected_share,
            "scenario_count": scenario_count,
            "top_scenarios": json.dumps(dict(scenario_counter.most_common(5)), ensure_ascii=False),
            "shared_device_count": int(type_counts.get("device", 0)),
            "shared_ip_count": int(type_counts.get("ip", 0)),
            "shared_merchant_count": int(type_counts.get("merchant", 0)),
            "shared_payee_count": int(type_counts.get("payee", 0)),
            "evidence_count": int(len(evidence)),
            "sample_users": json.dumps(users[:12], ensure_ascii=False),
            "sample_resources": json.dumps(sorted([m for m in members if m not in payer_set])[:12], ensure_ascii=False),
        }
        evidence_payload = [
            {
                "entity_id": item.get("entity_id"),
                "entity_type": item.get("entity_type"),
                "evidence_kind": item.get("evidence_kind"),
                "shared_user_count": int(item.get("shared_user_count", 0)),
                "txn_count": int(item.get("txn_count", 0)),
                "fraud_seed_count": int(item.get("fraud_seed_count", 0)),
                "resource_risk_score": float(item.get("resource_risk_score", 0.0)),
            }
            for item in evidence[:20]
        ]
        explanations = group_explanations(group, evidence_payload)
        group["risk_level"] = explanations["risk_level"]
        group["explanation_codes"] = json.dumps(explanations["explanation_codes"], ensure_ascii=False)
        group["explanation_text"] = explanations["explanation_text"]
        group["evidence_summary"] = json.dumps(explanations["evidence_summary"], ensure_ascii=False)
        group_rows.append(group)
        feature_row = {
            "fraud_group_mining_id": group_id,
            "graph_mining_group_risk_score": risk_score,
            "graph_mining_group_entity_count": int(len(members)),
            "graph_mining_group_user_count": int(len(users)),
            "graph_mining_group_resource_count": int(len(members) - len(users)),
            "graph_mining_fraud_seed_count": int(component_tx["is_fraud"].sum()),
            "graph_mining_fraud_seed_ratio": float(component_tx["is_fraud"].sum() / max(len(component_tx), 1)),
            "graph_mining_shared_device_count": int(type_counts.get("device", 0)),
            "graph_mining_shared_ip_count": int(type_counts.get("ip", 0)),
            "graph_mining_shared_merchant_count": int(type_counts.get("merchant", 0)),
            "graph_mining_shared_payee_count": int(type_counts.get("payee", 0)),
            "graph_mining_evidence_count": int(len(evidence)),
            "graph_mining_scenario_count": scenario_count,
        }
        for entity_id in members:
            entity_rows.append({"entity_id": entity_id, **feature_row})
        evidence_rows.append({
            "fraud_group_mining_id": group_id,
            "risk_score": risk_score,
            "risk_level": explanations["risk_level"],
            "explanation_codes": explanations["explanation_codes"],
            "explanation_text": explanations["explanation_text"],
            "evidence_summary": explanations["evidence_summary"],
            "top_scenarios": dict(scenario_counter.most_common(5)),
            "sample_users": users[:12],
            "sample_resources": sorted([m for m in members if m not in payer_set])[:12],
            "evidence": evidence_payload,
            "subgraph": build_group_subgraph(group, evidence_payload),
        })

    groups = pd.DataFrame(group_rows)
    entity_features = pd.DataFrame(entity_rows)
    if entity_features.empty:
        entity_features = pd.DataFrame(columns=["entity_id", *ENTITY_FEATURE_COLUMNS])
    else:
        entity_features = entity_features.drop_duplicates("entity_id", keep="first")

    summary = evaluate_graph_mining(tx_eval, entity_features, groups)
    summary["threshold_breakdown"] = threshold_breakdown(tx_eval, entity_features, groups)
    summary["recommended_high_confidence_threshold"] = 0.65
    summary["high_confidence"] = summary["threshold_breakdown"].get("risk_ge_0.65", {})
    summary.update({
        "dataset": ds_dir.name,
        "mining_splits": list(mining_splits),
        "evaluation_splits": list(evaluation_splits),
        "candidate_resource_count": int(len(resources)),
        "detected_group_count": int(len(groups)),
        "detected_entity_count": int(len(entity_features)),
        "min_group_risk_score": float(min_group_risk_score),
        "max_users_per_resource": int(max_users_per_resource),
    })

    out_dir = mining_dir(ds_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entity_features.to_parquet(out_entity, index=False)
    groups.to_parquet(out_groups, index=False)
    with evidence_path(ds_dir).open("w", encoding="utf-8") as f:
        for row in evidence_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return entity_features, groups, summary


def evaluate_graph_mining(tx_eval: pd.DataFrame, entity_features: pd.DataFrame, groups: pd.DataFrame) -> dict[str, Any]:
    if tx_eval.empty or entity_features.empty:
        return {
            "injected_row_recall": 0.0,
            "injected_group_recall": 0.0,
            "flagged_row_precision_vs_injection": 0.0,
            "average_detected_group_purity": 0.0,
        }
    mapper = entity_features[["entity_id", "fraud_group_mining_id", "graph_mining_group_risk_score"]].rename(
        columns={"entity_id": "payer_id"}
    )
    rows = tx_eval.merge(mapper, on="payer_id", how="left")
    rows["is_detected_by_graph"] = rows["fraud_group_mining_id"].notna()
    injected = rows[rows["is_injected"] == 1]
    flagged = rows[rows["is_detected_by_graph"]]
    injected_groups = set(injected["fraud_group_id"].dropna().astype(str))
    detected_injected_groups = set(flagged.loc[flagged["is_injected"] == 1, "fraud_group_id"].dropna().astype(str))
    purities = []
    if not groups.empty:
        for _, group in groups.iterrows():
            group_rows = flagged[flagged["fraud_group_mining_id"] == group["fraud_group_mining_id"]]
            group_injected = group_rows["fraud_group_id"].dropna().astype(str)
            if not group_injected.empty:
                purities.append(float(group_injected.value_counts().iloc[0] / len(group_injected)))
    return {
        "injected_rows": int(len(injected)),
        "detected_injected_rows": int(injected["is_detected_by_graph"].sum()) if not injected.empty else 0,
        "injected_row_recall": float(injected["is_detected_by_graph"].mean()) if not injected.empty else 0.0,
        "injected_group_count": int(len(injected_groups)),
        "detected_injected_group_count": int(len(detected_injected_groups)),
        "injected_group_recall": float(len(detected_injected_groups) / max(len(injected_groups), 1)),
        "flagged_rows": int(len(flagged)),
        "flagged_injected_rows": int((flagged["is_injected"] == 1).sum()),
        "flagged_row_precision_vs_injection": float((flagged["is_injected"] == 1).mean()) if not flagged.empty else 0.0,
        "average_detected_group_purity": float(sum(purities) / len(purities)) if purities else 0.0,
    }


def threshold_breakdown(
    tx_eval: pd.DataFrame,
    entity_features: pd.DataFrame,
    groups: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.35, 0.55, 0.65, 0.75),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if entity_features.empty:
        return out
    for threshold in thresholds:
        filtered_entities = entity_features[
            pd.to_numeric(entity_features["graph_mining_group_risk_score"], errors="coerce").fillna(0.0) >= threshold
        ]
        filtered_groups = groups[
            pd.to_numeric(groups["graph_mining_group_risk_score"], errors="coerce").fillna(0.0) >= threshold
        ] if not groups.empty else groups
        out[f"risk_ge_{threshold:.2f}"] = evaluate_graph_mining(tx_eval, filtered_entities, filtered_groups)
    return out


def load_entity_graph_risk(ds_dir: Path, force: bool = False) -> pd.DataFrame:
    entity_features, _, _ = mine_fraud_graph(ds_dir, force=force)
    return entity_features


def _load_evidence_rows(ds_dir: Path) -> dict[str, dict[str, Any]]:
    path = evidence_path(ds_dir)
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        mine_fraud_graph(ds_dir)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            group_id = row.get("fraud_group_mining_id")
            if group_id:
                rows[str(group_id)] = row
    return rows


def load_group_detail(ds_dir: Path, group_id: str) -> dict[str, Any] | None:
    if not groups_path(ds_dir).exists():
        mine_fraud_graph(ds_dir)
    if not groups_path(ds_dir).exists():
        return None
    groups = pd.read_parquet(groups_path(ds_dir))
    matched = groups[groups["fraud_group_mining_id"] == group_id]
    if matched.empty:
        return None
    group = matched.where(pd.notna(matched), None).iloc[0].to_dict()
    evidence_row = _load_evidence_rows(ds_dir).get(group_id, {})
    evidence = evidence_row.get("evidence") or []
    explanations = group_explanations(group, evidence)
    subgraph = evidence_row.get("subgraph") or build_group_subgraph(group, evidence)
    return {
        "group": group,
        "evidence": evidence,
        "explanations": {
            "risk_level": evidence_row.get("risk_level") or explanations["risk_level"],
            "explanation_codes": evidence_row.get("explanation_codes") or explanations["explanation_codes"],
            "explanation_text": evidence_row.get("explanation_text") or explanations["explanation_text"],
            "evidence_summary": evidence_row.get("evidence_summary") or explanations["evidence_summary"],
        },
        "subgraph": subgraph,
    }


def load_entity_group_detail(ds_dir: Path, entity_id: str) -> dict[str, Any] | None:
    if not entity_risk_path(ds_dir).exists():
        mine_fraud_graph(ds_dir)
    if not entity_risk_path(ds_dir).exists():
        return None
    features = pd.read_parquet(entity_risk_path(ds_dir), columns=["entity_id", "fraud_group_mining_id"])
    matched = features[features["entity_id"] == entity_id]
    if matched.empty:
        return None
    group_id = matched.iloc[0]["fraud_group_mining_id"]
    if pd.isna(group_id):
        return None
    detail = load_group_detail(ds_dir, str(group_id))
    if detail is None:
        return None
    detail["matched_entity_id"] = entity_id
    return detail


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ds_dir = dataset_dir(config, args.dataset)
    entity_features, groups, summary = mine_fraud_graph(
        ds_dir,
        force=args.force,
        max_users_per_resource=args.max_users_per_resource,
        min_group_risk_score=args.min_group_risk_score,
    )
    print(json.dumps({
        "entity_features": len(entity_features),
        "groups": len(groups),
        "summary": summary,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine fraud rings from FP-FraudSim heterogeneous graph signals.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-users-per-resource", type=int, default=200)
    parser.add_argument("--min-group-risk-score", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
