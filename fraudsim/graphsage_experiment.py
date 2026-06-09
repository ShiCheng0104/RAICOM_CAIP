from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fraudsim.config import dataset_dir, load_config


def sample_graph(edges: pd.DataFrame, max_edges: int, max_nodes: int, seed: int) -> pd.DataFrame:
    labeled = edges[edges["label"].isin([0, 1])].copy()
    positives = labeled[labeled["label"] == 1]
    negatives = labeled[labeled["label"] == 0]
    # Keep fraud edges visible without letting them dominate the sampled graph.
    positive_limit = min(len(positives), max_edges // 5)
    negative_limit = min(len(negatives), max_edges - positive_limit)
    sampled = pd.concat([
        positives.sample(positive_limit, random_state=seed) if len(positives) > positive_limit else positives,
        negatives.sample(negative_limit, random_state=seed) if len(negatives) > negative_limit else negatives,
    ]).sample(frac=1.0, random_state=seed)

    mixed_nodes = pd.unique(sampled[["src_id", "dst_id"]].to_numpy().ravel())
    selected_nodes = set(mixed_nodes[:max_nodes])
    return sampled[sampled["src_id"].isin(selected_nodes) & sampled["dst_id"].isin(selected_nodes)].copy()


def build_graph_arrays(edges: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    node_ids = pd.unique(pd.concat([edges["src_id"], edges["dst_id"]], ignore_index=True)).tolist()
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    src = edges["src_id"].map(node_index).to_numpy(dtype=np.int64)
    dst = edges["dst_id"].map(node_index).to_numpy(dtype=np.int64)
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])

    node_type: dict[str, str] = {}
    node_type.update(dict(zip(edges["src_id"], edges["src_type"])))
    node_type.update(dict(zip(edges["dst_id"], edges["dst_type"])))
    types = sorted(set(node_type.values()))
    type_index = {name: idx for idx, name in enumerate(types)}
    degree = np.bincount(edge_index[0], minlength=len(node_ids)).astype(np.float32)
    features = np.zeros((len(node_ids), len(types) + 2), dtype=np.float32)
    for idx, node_id in enumerate(node_ids):
        features[idx, type_index[node_type[node_id]]] = 1.0
    features[:, -2] = np.log1p(degree)
    features[:, -1] = degree / max(float(degree.max()), 1.0)

    labels = np.zeros(len(node_ids), dtype=np.int64)
    fraud_edges = edges[edges["label"] == 1]
    fraud_nodes = set(fraud_edges["src_id"]) | set(fraud_edges["dst_id"])
    for node_id in fraud_nodes:
        if node_id in node_index:
            labels[node_index[node_id]] = 1
    return node_ids, edge_index, features, labels


def run_experiment(
    ds_dir: Path,
    output_dir: Path,
    max_edges: int = 80000,
    max_nodes: int = 30000,
    epochs: int = 20,
    hidden_dim: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("GraphSAGE experiment requires PyTorch.") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    raw_edges = pd.read_parquet(
        ds_dir / "graph_edges.parquet",
        columns=["src_id", "dst_id", "src_type", "dst_type", "label"],
    )
    edges = sample_graph(raw_edges, max_edges=max_edges, max_nodes=max_nodes, seed=seed)
    node_ids, edge_index_np, features_np, labels_np = build_graph_arrays(edges)
    indices = np.arange(len(node_ids))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=seed,
        stratify=labels_np,
    )
    train_idx, valid_idx = train_test_split(
        train_idx,
        test_size=0.20,
        random_state=seed,
        stratify=labels_np[train_idx],
    )

    x = torch.tensor(features_np, dtype=torch.float32)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    labels = torch.tensor(labels_np, dtype=torch.float32)

    class GraphSAGELayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim * 2, out_dim)

        def forward(self, node_features: Any, graph_edges: Any) -> Any:
            source, target = graph_edges
            neighbor_sum = torch.zeros_like(node_features)
            neighbor_sum.index_add_(0, target, node_features[source])
            degree = torch.zeros(node_features.shape[0], dtype=node_features.dtype)
            degree.index_add_(0, target, torch.ones_like(target, dtype=node_features.dtype))
            neighbor_mean = neighbor_sum / degree.clamp(min=1).unsqueeze(1)
            return torch.relu(self.linear(torch.cat([node_features, neighbor_mean], dim=1)))

    class GraphSAGE(nn.Module):
        def __init__(self, input_dim: int, embedding_dim: int) -> None:
            super().__init__()
            self.layer1 = GraphSAGELayer(input_dim, embedding_dim)
            self.layer2 = GraphSAGELayer(embedding_dim, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, 1)

        def forward(self, node_features: Any, graph_edges: Any) -> tuple[Any, Any]:
            embedding = self.layer2(self.layer1(node_features, graph_edges), graph_edges)
            return embedding, self.classifier(embedding).squeeze(1)

    model = GraphSAGE(x.shape[1], hidden_dim)
    positive_weight = float((labels_np == 0).sum() / max((labels_np == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    valid_tensor = torch.tensor(valid_idx, dtype=torch.long)
    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_valid_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        _, logits = model(x, edge_index)
        loss = criterion(logits[train_tensor], labels[train_tensor])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, valid_logits = model(x, edge_index)
            valid_loss = float(criterion(valid_logits[valid_tensor], labels[valid_tensor]).item())
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "valid_loss": valid_loss})
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeddings, logits = model(x, edge_index)
        scores = torch.sigmoid(logits).numpy()
    test_labels = labels_np[test_idx]
    test_scores = scores[test_idx]
    predictions = (test_scores >= 0.5).astype(int)
    metrics = {
        "experiment": "graphsage_sidecar",
        "dataset": ds_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampled_edges": int(len(edges)),
        "sampled_nodes": int(len(node_ids)),
        "fraud_node_ratio": float(labels_np.mean()),
        "epochs": int(epochs),
        "embedding_dim": int(hidden_dim),
        "test_rows": int(len(test_idx)),
        "pr_auc": float(average_precision_score(test_labels, test_scores)),
        "roc_auc": float(roc_auc_score(test_labels, test_scores)),
        "precision_at_0_5": float(precision_score(test_labels, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(test_labels, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(test_labels, predictions, zero_division=0)),
        "best_valid_loss": best_valid_loss,
        "role": "sidecar_experiment_not_online_decision",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_frame = pd.DataFrame(
        embeddings.numpy(),
        columns=[f"graphsage_embedding_{idx:02d}" for idx in range(hidden_dim)],
    )
    embedding_frame.insert(0, "node_id", node_ids)
    embedding_frame["graphsage_risk_score"] = scores
    embedding_frame["graphsage_label"] = labels_np
    embedding_frame.to_parquet(output_dir / "node_embeddings.parquet", index=False)
    torch.save({"model_state": model.state_dict(), "input_dim": x.shape[1], "embedding_dim": hidden_dim}, output_dir / "model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight GraphSAGE sidecar experiment.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default="models/graphsage_sidecar/latest")
    parser.add_argument("--max-edges", type=int, default=80000)
    parser.add_argument("--max-nodes", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = run_experiment(
        dataset_dir(config, args.dataset),
        Path(args.output),
        max_edges=args.max_edges,
        max_nodes=args.max_nodes,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
