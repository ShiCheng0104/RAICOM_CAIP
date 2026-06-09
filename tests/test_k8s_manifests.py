from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class KubernetesManifestTests(unittest.TestCase):
    def test_edge_manifest_contains_runtime_and_security_resources(self) -> None:
        resources = list(yaml.safe_load_all(Path("deploy/k8s/edge.yaml").read_text(encoding="utf-8")))
        kinds = [resource["kind"] for resource in resources]
        names = {resource["metadata"]["name"] for resource in resources}
        self.assertIn("NetworkPolicy", kinds)
        self.assertIn("Secret", kinds)
        self.assertTrue({"kafka", "redis", "model-api", "flink-jobmanager", "flink-taskmanager", "flink-risk-job"} <= names)
        self.assertGreaterEqual(kinds.count("PersistentVolumeClaim"), 2)

    def test_center_manifest_contains_iteration_jobs(self) -> None:
        resources = list(yaml.safe_load_all(Path("deploy/k8s/center.yaml").read_text(encoding="utf-8")))
        cronjobs = {resource["metadata"]["name"] for resource in resources if resource["kind"] == "CronJob"}
        self.assertEqual(cronjobs, {"candidate-model-training", "graphsage-sidecar-training"})
        self.assertGreaterEqual([resource["kind"] for resource in resources].count("PersistentVolumeClaim"), 2)


if __name__ == "__main__":
    unittest.main()
