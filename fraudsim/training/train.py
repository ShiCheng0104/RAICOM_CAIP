from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib

from fraudsim.config import dataset_dir, load_config, model_dir
from fraudsim.features import apply_feature_config, enrich_transactions, load_split, select_training_frame
from fraudsim.graph_features import add_graph_features
from fraudsim.models import get_model_adapter
from fraudsim.window_features import add_offline_window_features


def recall_at_top_k(y_true: pd.Series, scores: pd.Series, ratio: float = 0.01) -> float:
    if len(y_true) == 0 or int(y_true.sum()) == 0:
        return 0.0
    k = max(1, int(len(y_true) * ratio))
    top_idx = np.argsort(-scores.to_numpy())[:k]
    return float(y_true.iloc[top_idx].sum() / y_true.sum())


def false_positive_rate(y_true: pd.Series, pred: np.ndarray) -> float:
    negatives = y_true == 0
    if int(negatives.sum()) == 0:
        return 0.0
    return float(((pred == 1) & negatives.to_numpy()).sum() / negatives.sum())


def build_metrics(
    model_name: str,
    dataset: str,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    y_test: pd.Series,
    scores: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int).to_numpy()
    return {
        "model_name": model_name,
        "dataset": dataset,
        "train_rows": int(train_rows),
        "valid_rows": int(valid_rows),
        "test_rows": int(test_rows),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)) if len(set(y_test)) > 1 else 0.0,
        "f1": float(f1_score(y_test, pred)),
        "recall_at_top_1pct": recall_at_top_k(y_test, scores, 0.01),
        "false_positive_rate_at_threshold": false_positive_rate(y_test, pred),
        "thresholds": {
            "medium": 0.50,
            "high": threshold,
        },
    }


def threshold_report(y_true: pd.Series, scores: pd.Series, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int).to_numpy()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": false_positive_rate(y_true, pred),
        "predicted_positive": int(pred.sum()),
    }


def calibrate_thresholds(y_valid: pd.Series, valid_scores: pd.Series) -> dict[str, Any]:
    candidates = np.linspace(0.01, 0.99, 99)
    reports = [threshold_report(y_valid, valid_scores, float(threshold)) for threshold in candidates]
    best_f1 = max(reports, key=lambda row: row["f1"])
    high_precision = [row for row in reports if row["precision"] >= 0.95]
    high_recall = [row for row in reports if row["recall"] >= 0.90]
    return {
        "best_f1": best_f1,
        "precision_at_least_0_95": max(high_precision, key=lambda row: row["recall"]) if high_precision else None,
        "recall_at_least_0_90": max(high_recall, key=lambda row: row["precision"]) if high_recall else None,
        "default_medium": threshold_report(y_valid, valid_scores, 0.50),
        "default_high": threshold_report(y_valid, valid_scores, 0.80),
    }


def update_leaderboard(models_root: Path, metrics: dict[str, Any]) -> None:
    leaderboard_path = models_root / "leaderboard.json"
    if leaderboard_path.exists():
        with leaderboard_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        rows = []

    rows = [
        row for row in rows
        if not (row.get("model_name") == metrics["model_name"] and row.get("dataset") == metrics["dataset"])
    ]
    rows.append(metrics)
    rows.sort(key=lambda row: row.get("pr_auc", 0.0), reverse=True)

    models_root.mkdir(parents=True, exist_ok=True)
    with leaderboard_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def load_feedback(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    feedback_path = Path(path)
    if not feedback_path.exists():
        print(f"[fraudsim] feedback path not found: {feedback_path}")
        return pd.DataFrame()
    if feedback_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(feedback_path)
    else:
        rows = []
        with feedback_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    if "reviewed_is_fraud" in df.columns and "is_fraud" not in df.columns:
        df["is_fraud"] = df["reviewed_is_fraud"]
    if "label" in df.columns and "is_fraud" not in df.columns:
        df["is_fraud"] = df["label"]
    if "is_fraud" in df.columns:
        df = df[df["is_fraud"].isin([0, 1])].copy()
        df["is_fraud"] = df["is_fraud"].astype("int8")
    return df


def apply_feedback(all_raw: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    if feedback.empty or "is_fraud" not in feedback.columns:
        return all_raw
    out = all_raw.copy()
    feedback = feedback.copy()
    feedback["_fraudsim_split"] = "train"
    if "transaction_id" in feedback.columns and "transaction_id" in out.columns:
        label_map = feedback.dropna(subset=["transaction_id"]).drop_duplicates("transaction_id", keep="last")
        label_map = label_map.set_index("transaction_id")["is_fraud"]
        matched = out["transaction_id"].isin(label_map.index)
        out.loc[matched, "is_fraud"] = out.loc[matched, "transaction_id"].map(label_map).astype("int8")
        new_feedback = feedback[~feedback["transaction_id"].isin(out["transaction_id"])]
    else:
        new_feedback = feedback
    if not new_feedback.empty:
        for col in out.columns:
            if col not in new_feedback.columns:
                new_feedback[col] = pd.NA
        out = pd.concat([out, new_feedback[out.columns]], ignore_index=True)
    print(f"[fraudsim] feedback labeled rows used={len(feedback)}")
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    ds_dir = dataset_dir(config, args.dataset)
    adapter = get_model_adapter(args.model)
    out_dir = model_dir(config, args.model if args.output is None else None)
    if args.output:
        out_dir = Path(args.output)

    print(f"[fraudsim] dataset={ds_dir}")
    print(f"[fraudsim] model={args.model}")

    train_raw = load_split(ds_dir, "train")
    valid_raw = load_split(ds_dir, "valid")
    test_raw = load_split(ds_dir, "test")

    train_raw["_fraudsim_split"] = "train"
    valid_raw["_fraudsim_split"] = "valid"
    test_raw["_fraudsim_split"] = "test"
    all_raw = pd.concat([train_raw, valid_raw, test_raw], ignore_index=True)
    feedback_df = load_feedback(args.feedback_path)
    all_raw = apply_feedback(all_raw, feedback_df)
    if args.with_window_features:
        print("[fraudsim] adding offline window features")
        all_raw = add_offline_window_features(all_raw)
    if args.with_graph_features:
        print("[fraudsim] adding graph features")
        all_raw = add_graph_features(all_raw, ds_dir, force_rebuild=args.rebuild_graph_features)

    train_df = enrich_transactions(all_raw[all_raw["_fraudsim_split"] == "train"].drop(columns=["_fraudsim_split"]), ds_dir)
    valid_df = enrich_transactions(all_raw[all_raw["_fraudsim_split"] == "valid"].drop(columns=["_fraudsim_split"]), ds_dir)
    test_df = enrich_transactions(all_raw[all_raw["_fraudsim_split"] == "test"].drop(columns=["_fraudsim_split"]), ds_dir)

    x_train, y_train, feature_config = select_training_frame(train_df)
    x_valid = apply_feature_config(valid_df[valid_df["is_fraud"].isin([0, 1])], feature_config)
    y_valid = valid_df[valid_df["is_fraud"].isin([0, 1])]["is_fraud"].astype(int)
    x_test = apply_feature_config(test_df[test_df["is_fraud"].isin([0, 1])], feature_config)
    y_test = test_df[test_df["is_fraud"].isin([0, 1])]["is_fraud"].astype(int)

    artifact = adapter.fit(x_train, y_train, x_valid, y_valid)
    valid_scores = adapter.predict_proba(artifact, x_valid)
    test_scores = adapter.predict_proba(artifact, x_test)
    metrics = build_metrics(
        model_name=artifact.model_name,
        dataset=ds_dir.name,
        train_rows=len(x_train),
        valid_rows=len(x_valid),
        test_rows=len(x_test),
        y_test=y_test,
        scores=test_scores,
        threshold=args.high_threshold,
    )
    metrics["model_version"] = artifact.model_version
    metrics["feature_count"] = len(feature_config.feature_columns)
    metrics["with_window_features"] = bool(args.with_window_features)
    metrics["with_graph_features"] = bool(args.with_graph_features)
    metrics["feedback_rows"] = int(len(feedback_df))
    metrics["threshold_calibration"] = calibrate_thresholds(y_valid, valid_scores)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "artifact": {
                "model_name": artifact.model_name,
                "model_version": artifact.model_version,
            },
            "model": artifact.model,
            "feature_config": feature_config.to_dict(),
        },
        out_dir / "model.pkl",
    )
    with (out_dir / "feature_config.json").open("w", encoding="utf-8") as f:
        json.dump(feature_config.to_dict(), f, ensure_ascii=False, indent=2)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    update_leaderboard(Path("models"), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FP-FraudSim models.")
    parser.add_argument("--config", default=None, help="Path to fraudsim YAML config.")
    parser.add_argument("--dataset", default=None, help="Dataset folder name under data/processed.")
    parser.add_argument("--model", default="lightgbm", help="Model adapter name.")
    parser.add_argument("--output", default=None, help="Override output directory.")
    parser.add_argument("--high-threshold", type=float, default=0.80)
    parser.add_argument("--with-window-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-graph-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-graph-features", action="store_true")
    parser.add_argument("--feedback-path", default=None, help="Optional parquet/jsonl file with reviewed labels.")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
