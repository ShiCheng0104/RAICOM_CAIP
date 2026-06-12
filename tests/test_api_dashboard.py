from __future__ import annotations

import unittest

from fraudsim.api.app import (
    ReloadRequest,
    ThresholdSandboxRequest,
    dashboard,
    graphsage_metrics,
    health,
    leaderboard,
    list_models,
    metrics,
    reload_model,
    threshold_sandbox,
    threshold_sandbox_defaults,
    user_profile_detail,
    user_profiles,
)


class ApiDashboardTests(unittest.TestCase):
    def test_dashboard_asset_exists(self) -> None:
        response = dashboard()
        self.assertTrue(str(response.path).endswith("index.html"))
        self.assertIn("no-store", response.headers["cache-control"])

    def test_model_discovery_and_metrics_endpoints(self) -> None:
        models = list_models()
        self.assertIn("models", models)
        self.assertGreaterEqual(len(models["models"]), 1)
        self.assertIn("lightgbm", {row["name"] for row in models["models"]})

        self.assertIn("metrics", metrics())
        self.assertIn("rows", leaderboard())
        self.assertTrue(health()["loaded"])

    def test_reload_specific_model_directory(self) -> None:
        result = reload_model(ReloadRequest(model_name="lightgbm"))
        self.assertTrue(result["loaded"])
        self.assertEqual(result["model_name"], "lightgbm")

    def test_threshold_sandbox_and_graphsage_metrics(self) -> None:
        self.assertTrue(threshold_sandbox_defaults()["available"])
        result = threshold_sandbox(ThresholdSandboxRequest(medium_threshold=0.5, high_threshold=0.8))
        self.assertEqual(result["rows"], result["decisions"]["pass"] + result["decisions"]["review"] + result["decisions"]["reject"])
        self.assertTrue(graphsage_metrics()["available"])

    def test_user_profile_list_and_detail(self) -> None:
        result = user_profiles(limit=80)
        self.assertGreater(result["total"], 0)
        self.assertEqual(len(result["users"]), 80)
        self.assertEqual(result["sort"], "diverse")
        self.assertGreater(result["page_sources"], 1)
        self.assertGreater(result["page_profile_patterns"], 10)
        detail = user_profile_detail(result["users"][0]["user_id"])
        self.assertIn("profile", detail)
        self.assertEqual(detail["user_id"], detail["profile"]["user_id"])


if __name__ == "__main__":
    unittest.main()
