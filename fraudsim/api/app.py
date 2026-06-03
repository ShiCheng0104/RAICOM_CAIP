from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from confluent_kafka import Consumer, Producer, TopicPartition
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fraudsim.config import load_config, model_dir
from fraudsim.features import FeatureConfig, apply_feature_config, records_to_frame
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


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
STATIC_DIR = Path(__file__).resolve().parents[1] / "dashboard"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def kafka_bootstrap() -> str:
    return runtime.config.get("kafka", {}).get("bootstrap_servers", "kafka:9092")


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
    if score >= active_thresholds()["high"] and not reasons:
        reasons.append("model_high_score")
    return reasons[:5]


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
def reload_model(request: ReloadRequest | None = None) -> dict[str, Any]:
    request = request or ReloadRequest()
    runtime.load(model_name=request.model_name, model_path=request.model_path)
    if not runtime.loaded:
        raise HTTPException(status_code=404, detail=f"Model is not loaded from {runtime.path}")
    return health()


@app.post("/models/activate")
def activate_model(request: ReloadRequest) -> dict[str, Any]:
    return reload_model(request)


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
def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
    topic = runtime.config.get("topics", {}).get("feedback_events", "feedback_events")
    payload = request.model_dump()
    payload["feedback_created_at"] = datetime.now(timezone.utc).isoformat()
    producer = Producer({"bootstrap.servers": kafka_bootstrap()})
    producer.produce(
        topic=topic,
        key=request.transaction_id.encode("utf-8"),
        value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    producer.flush(10)
    return {"topic": topic, "written": True, "feedback": payload}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if not runtime.loaded:
        raise HTTPException(status_code=503, detail=f"Model is not loaded from {runtime.path}")

    records = request.records
    if records is None:
        records = [request.record] if request.record is not None else []
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
    return FileResponse(index_path)
