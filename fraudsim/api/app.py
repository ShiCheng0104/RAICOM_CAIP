from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from confluent_kafka import Consumer, Producer, TopicPartition
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fraudsim.config import load_config, model_dir
from fraudsim.config import dataset_dir as resolve_dataset_dir
from fraudsim.features import FeatureConfig, apply_feature_config, records_to_frame
from fraudsim.graph_mining import (
    entity_risk_path,
    groups_path,
    load_entity_group_detail,
    load_group_detail,
    mine_fraud_graph,
    summary_path,
)
from fraudsim.models.base import ModelArtifact
from fraudsim.models.registry import available_adapters, get_model_adapter


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] | None = Field(default=None)
    record: dict[str, Any] | None = Field(default=None)


class ReloadRequest(BaseModel):
    model_name: str | None = Field(default=None, description="Model directory under models/, for example lightgbm.")
    model_path: str | None = Field(default=None, description="Explicit path containing model.pkl.")


class FeedbackRequest(BaseModel):
    transaction_id: str = Field(description="Reviewed transaction id.")
    reviewed_is_fraud: int = Field(ge=0, le=1, description="Human reviewed label: 1 fraud, 0 normal.")
    reviewer: str | None = Field(default="dashboard")
    note: str | None = Field(default=None)
    source_topic: str | None = Field(default="risk_results")
    event: dict[str, Any] | None = Field(default=None)


class DemoRunRequest(BaseModel):
    action: str = Field(description="Demo action id.")


class PromoteModelRequest(BaseModel):
    model_name: str = Field(description="Model directory under models/.")


class AuditQuery(BaseModel):
    limit: int = 50


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = df.where(pd.notna(df), None).to_dict(orient="records")
    return rows


class ModelRuntime:
    def __init__(self) -> None:
        self.config = load_config()
        self.active_model = self.config.get("model", {}).get("active", "lightgbm")
        self.path = model_dir(self.config)
        self.adapter = get_model_adapter(self.active_model)
        self.artifact: ModelArtifact | None = None
        self.feature_config: FeatureConfig | None = None
        self.metrics: dict[str, Any] = {}
        self.loaded_at: str | None = None
        self.load()

    def resolve_model_path(self, model_name: str | None = None, model_path: str | None = None) -> Path:
        if model_path:
            return Path(model_path)
        if model_name:
            return Path("models") / model_name / "latest"
        return self.path

    def load(self, model_name: str | None = None, model_path: str | None = None) -> None:
        self.path = self.resolve_model_path(model_name=model_name, model_path=model_path)
        model_path = self.path / "model.pkl"
        if not model_path.exists():
            self.artifact = None
            self.feature_config = None
            self.metrics = {}
            self.loaded_at = None
            return
        payload = joblib.load(model_path)
        artifact_meta = payload.get("artifact", {})
        adapter_name = artifact_meta.get("model_name") or model_name or self.active_model
        self.adapter = get_model_adapter(adapter_name)
        self.active_model = adapter_name
        self.artifact = ModelArtifact(
            model_name=adapter_name,
            model_version=artifact_meta.get("model_version", "unknown"),
            model=payload["model"],
        )
        self.feature_config = FeatureConfig.from_dict(payload["feature_config"])
        self.metrics = _read_json(self.path / "metrics.json") or {}
        self.loaded_at = datetime.now(timezone.utc).isoformat()

    @property
    def loaded(self) -> bool:
        return self.artifact is not None and self.feature_config is not None


runtime = ModelRuntime()
app = FastAPI(title="FP-FraudSim Model API", version="0.1.0")
DEMO_RUNS: dict[str, dict[str, Any]] = {}
STATIC_DIR = Path(__file__).resolve().parents[1] / "dashboard"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def kafka_bootstrap() -> str:
    return runtime.config.get("kafka", {}).get("bootstrap_servers", "kafka:9092")


def require_api_key(request: Request = None) -> None:
    expected = os.getenv("FRAUDSIM_API_KEY")
    if not expected:
        return
    supplied = None
    if request is not None:
        supplied = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if supplied != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def audit_path() -> Path:
    return Path(os.getenv("FRAUDSIM_AUDIT_LOG", "models/audit/audit.jsonl"))


def audit_event(action: str, status: str, detail: dict[str, Any] | None = None, request: Request = None) -> None:
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "status": status,
        "detail": detail or {},
        "client": request.client.host if request and request.client else None,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def feedback_pool_path() -> Path:
    ds_dir = active_dataset_dir()
    return ds_dir / "feedback" / "feedback_pool.jsonl"


def append_feedback_pool(payload: dict[str, Any]) -> Path:
    path = feedback_pool_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def demo_actions() -> dict[str, dict[str, Any]]:
    dataset = runtime.config.get("dataset", {}).get("name", "fp_fraudsim_injected")
    bootstrap = kafka_bootstrap()
    return {
        "load_profiles": {
            "label": "初始化画像",
            "description": "把用户、商户、设备、IP 和图统计画像加载到 Redis。",
            "command": [
                sys.executable,
                "-m",
                "fraudsim.streaming.load_profiles",
                "--dataset",
                dataset,
                "--redis-url",
                "redis://redis:6379/0",
            ],
        },
        "replay_stream": {
            "label": "回放交易流",
            "description": "向 Kafka transaction_events 写入一小段交易流，用于观察 Flink 实时输出。",
            "command": [
                sys.executable,
                "-m",
                "fraudsim.streaming.producer",
                "--dataset",
                dataset,
                "--bootstrap-servers",
                bootstrap,
                "--rate",
                "20",
                "--limit",
                "300",
            ],
        },
        "simulate_stream": {
            "label": "生成仿真交易流",
            "description": "按默认参数生成一段带团伙欺诈的可调参仿真交易流。",
            "command": [
                sys.executable,
                "-m",
                "fraudsim.simulator.generate",
                "--dataset",
                dataset,
                "--rows",
                "5000",
                "--fraud-ratio",
                "0.04",
            ],
        },
        "retrain_logistic": {
            "label": "轻量重训",
            "description": "运行一个轻量 sklearn_logistic 重训演示，产物进入 models/sklearn_logistic/latest。",
            "command": [
                sys.executable,
                "-m",
                "fraudsim.training.train",
                "--dataset",
                dataset,
                "--model",
                "sklearn_logistic",
            ],
        },
        "adaptive_retrain": {
            "label": "反馈重训候选模型",
            "description": "使用人工审核反馈池训练 candidate 模型，等待评估后发布。",
            "command": [
                sys.executable,
                "-m",
                "fraudsim.training.train_with_feedback",
                "--dataset",
                dataset,
                "--model",
                "lightgbm",
                "--feedback-path",
                str(feedback_pool_path()),
            ],
        },
        "graph_mining": {
            "label": "图挖掘团伙",
            "description": "从共享设备、IP、商户、收款方和历史欺诈种子中挖掘疑似团伙。",
            "command": [
                sys.executable,
                "-m",
                "fraudsim.graph_mining",
                "--dataset",
                dataset,
                "--force",
            ],
        },
    }


def active_thresholds() -> dict[str, float]:
    calibration = runtime.metrics.get("threshold_calibration") or {}
    high_row = calibration.get("precision_at_least_0_95") or calibration.get("default_high") or {}
    medium_row = calibration.get("recall_at_least_0_90") or calibration.get("default_medium") or {}
    thresholds = runtime.metrics.get("thresholds") or {}
    high = float(high_row.get("threshold", thresholds.get("high", 0.80)))
    medium = float(medium_row.get("threshold", thresholds.get("medium", 0.50)))
    if medium >= high:
        medium = float(thresholds.get("medium", 0.50))
    return {"medium": medium, "high": high}


def decision(score: float) -> tuple[str, str]:
    thresholds = active_thresholds()
    if score >= thresholds["high"]:
        return "high", "reject"
    if score >= thresholds["medium"]:
        return "medium", "review"
    return "low", "pass"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def reason_codes(record: dict[str, Any], score: float) -> list[str]:
    reasons: list[str] = []
    window = record.get("window_features") or {}
    user_profile = record.get("user_profile") or {}
    device_profile = record.get("device_profile") or {}
    ip_profile = record.get("ip_profile") or {}
    merchant_profile = record.get("merchant_profile") or {}
    graph = record.get("graph_features") or {}

    if _num(record.get("amount")) >= max(_num(user_profile.get("avg_amount")) * 3, 5000):
        reasons.append("amount_deviation")
    if _num(window.get("user_txn_count_5min")) >= 5 or _num(window.get("user_amount_sum_5min")) >= 10000:
        reasons.append("high_frequency_user_window")
    if _num(window.get("device_unique_user_count_10min")) >= 3 or _num(device_profile.get("bind_user_count")) >= 10:
        reasons.append("shared_device")
    if _num(window.get("ip_unique_user_count_10min")) >= 5 or _num(ip_profile.get("is_proxy")) or _num(ip_profile.get("is_vpn")):
        reasons.append("risky_ip")
    if _num(window.get("merchant_unique_user_count_1h")) >= 20 or _num(merchant_profile.get("complaint_rate")) >= 0.05:
        reasons.append("merchant_concentration")
    if _num(graph.get("payer_graph_fraud_edge_ratio")) >= 0.2 or _num(graph.get("payer_graph_fraud_edge_count")) >= 3:
        reasons.append("graph_fraud_neighborhood")
    if _num(graph.get("payer_graph_mining_group_risk_score")) >= 0.70:
        reasons.append("fraud_ring_detected")
    if _num(graph.get("payer_graph_mining_shared_device_count")) >= 2:
        reasons.append("shared_device_ring")
    if _num(graph.get("payer_graph_mining_shared_ip_count")) >= 2:
        reasons.append("shared_ip_cluster")
    if _num(graph.get("merchant_graph_mining_group_risk_score")) >= 0.60:
        reasons.append("merchant_fraud_community")
    if score >= active_thresholds()["high"] and not reasons:
        reasons.append("model_high_score")
    return reasons[:5]


def active_dataset_dir() -> Path:
    return resolve_dataset_dir(runtime.config, runtime.config.get("dataset", {}).get("name"))


def _read_log_tail(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def demo_run_status(run_id: str) -> dict[str, Any]:
    row = DEMO_RUNS.get(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Demo run not found: {run_id}")
    process: subprocess.Popen | None = row.get("process")
    exit_code = process.poll() if process is not None else row.get("exit_code")
    status = "running" if exit_code is None else "succeeded" if exit_code == 0 else "failed"
    if exit_code is not None:
        row["exit_code"] = exit_code
        if not row.get("finished_at"):
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
    started_ts = row.get("started_at_ts")
    elapsed_seconds = None
    if started_ts is not None:
        elapsed_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - float(started_ts))
    log_path = Path(row["log_path"])
    return {
        "run_id": run_id,
        "action": row["action"],
        "label": row.get("label"),
        "description": row.get("description"),
        "command": row.get("command"),
        "pid": process.pid if process is not None else None,
        "status": status,
        "exit_code": exit_code,
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "elapsed_seconds": elapsed_seconds,
        "log_path": str(log_path),
        "log_tail": _read_log_tail(log_path),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "loaded": runtime.loaded,
        "model_name": runtime.artifact.model_name if runtime.artifact else runtime.active_model,
        "model_version": runtime.artifact.model_version if runtime.artifact else None,
        "feature_count": len(runtime.feature_config.feature_columns) if runtime.feature_config else 0,
        "model_path": str(runtime.path),
        "thresholds": active_thresholds() if runtime.loaded else None,
        "loaded_at": runtime.loaded_at,
    }


@app.post("/reload")
def reload_model(payload: ReloadRequest = None, request: Request = None) -> dict[str, Any]:
    require_api_key(request)
    payload = payload or ReloadRequest()
    runtime.load(model_name=payload.model_name, model_path=payload.model_path)
    if not runtime.loaded:
        audit_event("reload_model", "failed", {"model_path": str(runtime.path)}, request)
        raise HTTPException(status_code=404, detail=f"Model is not loaded from {runtime.path}")
    audit_event("reload_model", "succeeded", {"model_path": str(runtime.path)}, request)
    return health()


@app.post("/models/activate")
def activate_model(payload: ReloadRequest, request: Request = None) -> dict[str, Any]:
    return reload_model(payload, request)


def _copy_model_dir(src: Path, dst: Path) -> None:
    if not src.exists() or not (src / "model.pkl").exists():
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


@app.get("/models/{model_name}/candidate")
def model_candidate(model_name: str) -> dict[str, Any]:
    path = Path("models") / model_name / "candidate"
    metrics = _read_json(path / "metrics.json")
    manifest = _read_json(path / "candidate_manifest.json")
    return {
        "model_name": model_name,
        "exists": bool((path / "model.pkl").exists()),
        "path": str(path),
        "metrics": metrics,
        "manifest": manifest,
    }


@app.post("/models/promote")
def promote_model(payload: PromoteModelRequest, request: Request = None) -> dict[str, Any]:
    require_api_key(request)
    root = Path("models") / payload.model_name
    candidate = root / "candidate"
    latest = root / "latest"
    rollback = root / "rollback"
    if not candidate.exists() or not (candidate / "model.pkl").exists():
        raise HTTPException(status_code=404, detail=f"Candidate model not found: {candidate}")
    if latest.exists():
        _copy_model_dir(latest, rollback)
    _copy_model_dir(candidate, latest)
    runtime.load(model_name=payload.model_name)
    audit_event("promote_model", "succeeded", {"model_name": payload.model_name}, request)
    return {"promoted": True, "model_name": payload.model_name, "active": health()}


@app.post("/models/rollback")
def rollback_model(payload: PromoteModelRequest, request: Request = None) -> dict[str, Any]:
    require_api_key(request)
    root = Path("models") / payload.model_name
    rollback = root / "rollback"
    latest = root / "latest"
    _copy_model_dir(rollback, latest)
    runtime.load(model_name=payload.model_name)
    audit_event("rollback_model", "succeeded", {"model_name": payload.model_name}, request)
    return {"rolled_back": True, "model_name": payload.model_name, "active": health()}


@app.get("/models/adapters")
def model_adapters() -> dict[str, Any]:
    return {"adapters": available_adapters()}


@app.get("/models")
def list_models() -> dict[str, Any]:
    models_root = Path("models")
    rows = []
    if models_root.exists():
        for model_file in models_root.glob("*/latest/model.pkl"):
            model_path = model_file.parent
            metrics = _read_json(model_path / "metrics.json") or {}
            feature_config = _read_json(model_path / "feature_config.json") or {}
            rows.append({
                "name": model_path.parent.name,
                "path": str(model_path),
                "loaded": model_path.resolve() == runtime.path.resolve() if runtime.path.exists() else str(model_path) == str(runtime.path),
                "model_name": metrics.get("model_name"),
                "model_version": metrics.get("model_version"),
                "dataset": metrics.get("dataset"),
                "feature_count": metrics.get("feature_count") or len(feature_config.get("feature_columns", [])),
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "f1": metrics.get("f1"),
                "with_window_features": metrics.get("with_window_features"),
                "with_graph_features": metrics.get("with_graph_features"),
                "with_graph_mining_features": metrics.get("with_graph_mining_features"),
                "updated_at": datetime.fromtimestamp(model_file.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
    rows.sort(key=lambda row: (row.get("loaded") is not True, row.get("name") or ""))
    return {
        "active": health(),
        "adapters": available_adapters(),
        "models": rows,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    active_metrics = runtime.metrics or _read_json(runtime.path / "metrics.json")
    eval_v2 = _read_json(Path("models/evaluation_summary_v2.json"))
    eval_v1 = _read_json(Path("models/evaluation_summary.json"))
    return {
        "active_model_path": str(runtime.path),
        "metrics": active_metrics,
        "evaluation_summary_v2": eval_v2,
        "evaluation_summary": eval_v1,
    }


@app.get("/leaderboard")
def leaderboard() -> dict[str, Any]:
    rows = _read_json(Path("models/leaderboard.json")) or []
    return {"rows": rows}


@app.get("/graph/mining/summary")
def graph_mining_summary(force: bool = False) -> dict[str, Any]:
    ds_dir = active_dataset_dir()
    if force or not summary_path(ds_dir).exists():
        _, _, summary = mine_fraud_graph(ds_dir, force=force)
        return summary
    return _read_json(summary_path(ds_dir)) or {}


@app.get("/graph/mining/groups")
def graph_mining_groups(limit: int = 20) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    ds_dir = active_dataset_dir()
    if not groups_path(ds_dir).exists():
        mine_fraud_graph(ds_dir)
    groups = pd.read_parquet(groups_path(ds_dir))
    groups = groups.sort_values("graph_mining_group_risk_score", ascending=False).head(limit)
    return {"groups": _json_records(groups)}


@app.get("/graph/mining/groups/{group_id}")
def graph_mining_group_detail(group_id: str) -> dict[str, Any]:
    ds_dir = active_dataset_dir()
    detail = load_group_detail(ds_dir, group_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Graph mining group not found: {group_id}")
    return detail


@app.get("/graph/mining/entity/{entity_id}")
def graph_mining_entity(entity_id: str) -> dict[str, Any]:
    ds_dir = active_dataset_dir()
    if not entity_risk_path(ds_dir).exists():
        mine_fraud_graph(ds_dir)
    features = pd.read_parquet(entity_risk_path(ds_dir))
    row = features[features["entity_id"] == entity_id]
    if row.empty:
        return {"entity_id": entity_id, "found": False}
    payload = _json_records(row.head(1))[0]
    return {"entity_id": entity_id, "found": True, "graph_mining": payload}


@app.get("/graph/mining/entity/{entity_id}/links")
def graph_mining_entity_links(entity_id: str) -> dict[str, Any]:
    ds_dir = active_dataset_dir()
    detail = load_entity_group_detail(ds_dir, entity_id)
    if detail is None:
        return {"entity_id": entity_id, "found": False}
    return {"entity_id": entity_id, "found": True, **detail}


@app.get("/topics/{topic}/recent")
def topic_recent(topic: str, limit: int = 20, timeout_ms: int = 800) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    consumer = Consumer({
        "bootstrap.servers": kafka_bootstrap(),
        "group.id": f"fraudsim-dashboard-{topic}-{datetime.now(timezone.utc).timestamp()}",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    try:
        metadata = consumer.list_topics(topic=topic, timeout=5)
        if topic not in metadata.topics or metadata.topics[topic].error is not None:
            raise HTTPException(status_code=404, detail=f"Kafka topic not found: {topic}")
        partitions = sorted(metadata.topics[topic].partitions)
        assignments: list[TopicPartition] = []
        for partition in partitions:
            tp = TopicPartition(topic, partition)
            low, high = consumer.get_watermark_offsets(tp, timeout=5)
            start = max(low, high - max(1, limit // max(1, len(partitions)) + 2))
            assignments.append(TopicPartition(topic, partition, start))
        consumer.assign(assignments)

        deadline = datetime.now(timezone.utc).timestamp() + timeout_ms / 1000
        rows: list[dict[str, Any]] = []
        while datetime.now(timezone.utc).timestamp() < deadline and len(rows) < limit * 2:
            msg = consumer.poll(0.1)
            if msg is None:
                continue
            if msg.error():
                continue
            raw = msg.value().decode("utf-8", errors="replace")
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            rows.append({
                "topic": msg.topic(),
                "partition": msg.partition(),
                "offset": msg.offset(),
                "key": msg.key().decode("utf-8", errors="replace") if msg.key() else None,
                "value": value,
            })
        rows.sort(key=lambda row: (row["partition"], row["offset"]))
        return {"topic": topic, "messages": rows[-limit:]}
    finally:
        consumer.close()


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest, request: Request = None) -> dict[str, Any]:
    require_api_key(request)
    topic = runtime.config.get("topics", {}).get("feedback_events", "feedback_events")
    row = payload.model_dump()
    row["feedback_created_at"] = datetime.now(timezone.utc).isoformat()
    pool_path = append_feedback_pool(row)
    producer = Producer({"bootstrap.servers": kafka_bootstrap()})
    producer.produce(
        topic=topic,
        key=payload.transaction_id.encode("utf-8"),
        value=json.dumps(row, ensure_ascii=False).encode("utf-8"),
    )
    producer.flush(10)
    audit_event("submit_feedback", "succeeded", {"transaction_id": payload.transaction_id, "pool_path": str(pool_path)}, request)
    return {"topic": topic, "written": True, "feedback_pool_path": str(pool_path), "feedback": row}


@app.get("/demo/actions")
def list_demo_actions() -> dict[str, Any]:
    actions = demo_actions()
    return {
        "actions": [
            {
                "id": action_id,
                "label": action["label"],
                "description": action["description"],
                "command": " ".join(action["command"]),
            }
            for action_id, action in actions.items()
        ]
    }


@app.post("/demo/run")
def run_demo_action(payload: DemoRunRequest, request: Request = None) -> dict[str, Any]:
    require_api_key(request)
    actions = demo_actions()
    action = actions.get(payload.action)
    if action is None:
        raise HTTPException(status_code=404, detail=f"Unknown demo action: {payload.action}")

    log_dir = Path("models/demo_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{payload.action}_{stamp}_{uuid.uuid4().hex[:8]}"
    log_path = log_dir / f"{run_id}.log"
    log_file = log_path.open("ab")
    try:
        process = subprocess.Popen(
            action["command"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
        )
    finally:
        log_file.close()
    started_at = datetime.now(timezone.utc)
    DEMO_RUNS[run_id] = {
        "process": process,
        "action": payload.action,
        "label": action["label"],
        "description": action["description"],
        "command": " ".join(action["command"]),
        "log_path": log_path,
        "started_at": started_at.isoformat(),
        "started_at_ts": started_at.timestamp(),
    }
    audit_event("run_demo_action", "started", {"action": payload.action, "run_id": run_id}, request)
    return {"started": True, **demo_run_status(run_id)}


@app.get("/demo/runs/{run_id}")
def get_demo_run(run_id: str) -> dict[str, Any]:
    return demo_run_status(run_id)


@app.get("/audit/recent")
def audit_recent(limit: int = 50, request: Request = None) -> dict[str, Any]:
    require_api_key(request)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    path = audit_path()
    if not path.exists():
        return {"path": str(path), "events": []}
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {"path": str(path), "events": rows[-limit:]}


@app.post("/predict")
def predict(payload: PredictRequest, request: Request = None) -> dict[str, Any]:
    require_api_key(request)
    if not runtime.loaded:
        raise HTTPException(status_code=503, detail=f"Model is not loaded from {runtime.path}")

    records = payload.records
    if records is None:
        records = [payload.record] if payload.record is not None else []
    if not records:
        raise HTTPException(status_code=400, detail="Provide either 'record' or non-empty 'records'.")

    frame = records_to_frame(records)
    x = apply_feature_config(frame, runtime.feature_config)
    scores = runtime.adapter.predict_proba(runtime.artifact, x)

    results = []
    scored_at = datetime.now(timezone.utc).isoformat()
    for record, score in zip(records, scores):
        risk_level, action = decision(float(score))
        results.append({
            "transaction_id": record.get("transaction_id"),
            "risk_score": float(score),
            "risk_level": risk_level,
            "decision": action,
            "reason_codes": reason_codes(record, float(score)),
            "model_name": runtime.artifact.model_name,
            "model_version": runtime.artifact.model_version,
            "thresholds": active_thresholds(),
            "scored_at": scored_at,
        })
    return {"results": results}


def create_app() -> FastAPI:
    return app


@app.get("/", include_in_schema=False)
def dashboard_root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard assets are not installed.")
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )
