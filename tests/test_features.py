from __future__ import annotations

import unittest

import pandas as pd

from fraudsim.features import FeatureConfig, apply_feature_config, records_to_frame


class FeatureTests(unittest.TestCase):
    def test_nested_profiles_are_flattened_to_training_like_columns(self) -> None:
        frame = records_to_frame([
            {
                "transaction_id": "tx1",
                "amount": 12.5,
                "user_profile": {"txn_count": 7, "home_country": "CN"},
                "device_profile": {"txn_count": 3, "device_type": "mobile"},
                "ip_profile": {"bind_user_count": 2, "is_vpn": 1},
                "merchant_profile": {"txn_count": 9, "merchant_category": "retail"},
                "window_features": {"user_txn_count_5min": 2},
                "graph_features": {"payer_graph_degree": 4},
            }
        ])

        self.assertEqual(frame.loc[0, "txn_count"], 7)
        self.assertEqual(frame.loc[0, "txn_count_device"], 3)
        self.assertEqual(frame.loc[0, "bind_user_count_ip"], 2)
        self.assertEqual(frame.loc[0, "txn_count_merchant"], 9)
        self.assertEqual(frame.loc[0, "merchant_category_merchant"], "retail")
        self.assertEqual(frame.loc[0, "window_features_user_txn_count_5min"], 2)
        self.assertEqual(frame.loc[0, "payer_graph_degree"], 4)

    def test_apply_feature_config_adds_missing_columns(self) -> None:
        config = FeatureConfig(
            feature_columns=["amount", "channel", "txn_count"],
            categorical_columns=["channel"],
            numeric_fill_values={"amount": 0.0, "txn_count": 5.0},
        )
        frame = pd.DataFrame([{"amount": "1.25"}])

        out = apply_feature_config(frame, config)

        self.assertEqual(list(out.columns), ["amount", "channel", "txn_count"])
        self.assertEqual(float(out.loc[0, "amount"]), 1.25)
        self.assertEqual(float(out.loc[0, "txn_count"]), 5.0)
        self.assertEqual(str(out.loc[0, "channel"]), "__missing__")


if __name__ == "__main__":
    unittest.main()
