from __future__ import annotations

import unittest

from fraudsim.api.app import dashboard, health, leaderboard, list_models, metrics, reload_model, ReloadRequest


class ApiDashboardTests(unittest.TestCase):
    def test_dashboard_asset_exists(self) -> None:
        response = dashboard()
        self.assertTrue(str(response.path).endswith("index.html"))

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


if __name__ == "__main__":
    unittest.main()
