from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudsim.config import dataset_dir, load_config, model_dir
from fraudsim.features import FeatureConfig, apply_feature_config, enrich_transactions, load_split
from fraudsim.graph_features import add_graph_features
from fraudsim.models.base import ModelArtifact
from fraudsim.models.registry import get_model_adapter
from fraudsim.window_features import add_offline_window_features


def build_scorecard(
    ds_dir: Path,
    artifact_dir: Path,
    with_window_features: bool = True,
    with_graph_features: bool = True,
    with_graph_mining_features: bool = False,
) -> dict[str, object]:
    payload = joblib.load(artifact_dir / "model.pkl")
    artifact_meta = payload.get("artifact", {})
    model_name = artifact_meta.get("model_name", artifact_dir.parent.name)
    artifact = ModelArtifact(
        model_name=model_name,
        model_version=artifact_meta.get("model_version", "unknown"),
        model=payload["model"],
    )
    feature_config = FeatureConfig.from_dict(payload["feature_config"])
    all_raw = pd.concat([
        load_split(ds_dir, "train").assign(_fraudsim_split="train"),
        load_split(ds_dir, "valid").assign(_fraudsim_split="valid"),
        load_split(ds_dir, "test").assign(_fraudsim_split="test"),
    ], ignore_index=True)
    if with_window_features:
        all_raw = add_offline_window_features(all_raw)
    if with_graph_features:
        all_raw = add_graph_features(all_raw, ds_dir, include_graph_mining=with_graph_mining_features)
    test_raw = all_raw[all_raw["_fraudsim_split"] == "test"].drop(columns=["_fraudsim_split"])
    test = enrich_transactions(test_raw, ds_dir)
    labeled = test[test["is_fraud"].isin([0, 1])]
    x_test = apply_feature_config(labeled, feature_config)
    scores = get_model_adapter(model_name).predict_proba(artifact, x_test)
    score_path = artifact_dir / "evaluation_scores.parquet"
    pd.DataFrame({
        "is_fraud": labeled["is_fraud"].to_numpy(dtype="int8"),
        "risk_score": np.asarray(scores, dtype="float32"),
    }).to_parquet(score_path, index=False)
    return {
        "model_name": model_name,
        "model_version": artifact.model_version,
        "rows": int(len(labeled)),
        "fraud_rows": int(labeled["is_fraud"].sum()),
        "score_path": str(score_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build threshold sandbox scorecard for an existing model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--with-window-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-graph-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-graph-mining-features", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifact_dir = Path(args.model_path) if args.model_path else model_dir(config, args.model)
    result = build_scorecard(
        dataset_dir(config, args.dataset),
        artifact_dir,
        with_window_features=args.with_window_features,
        with_graph_features=args.with_graph_features,
        with_graph_mining_features=args.with_graph_mining_features,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
