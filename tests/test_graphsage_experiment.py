from __future__ import annotations

import unittest

import pandas as pd

from fraudsim.graphsage_experiment import build_graph_arrays, sample_graph


class GraphSAGEExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = pd.DataFrame([
            {"src_id": "u1", "dst_id": "t1", "src_type": "user", "dst_type": "transaction", "label": 1},
            {"src_id": "u2", "dst_id": "t2", "src_type": "user", "dst_type": "transaction", "label": 0},
            {"src_id": "u3", "dst_id": "t3", "src_type": "user", "dst_type": "transaction", "label": 0},
        ])

    def test_builds_undirected_graph_and_node_labels(self) -> None:
        node_ids, edge_index, features, labels = build_graph_arrays(self.edges)
        self.assertEqual(edge_index.shape[1], len(self.edges) * 2)
        self.assertEqual(features.shape[0], len(node_ids))
        self.assertEqual(labels[node_ids.index("u1")], 1)
        self.assertEqual(labels[node_ids.index("u2")], 0)

    def test_sample_graph_respects_limits(self) -> None:
        sampled = sample_graph(self.edges, max_edges=2, max_nodes=4, seed=42)
        self.assertLessEqual(len(sampled), 2)
        self.assertLessEqual(len(set(sampled["src_id"]) | set(sampled["dst_id"])), 4)


if __name__ == "__main__":
    unittest.main()
