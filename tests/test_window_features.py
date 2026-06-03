from __future__ import annotations

import unittest

import pandas as pd

from fraudsim.window_features import add_offline_window_features


class WindowFeatureTests(unittest.TestCase):
    def test_offline_window_features_follow_event_time(self) -> None:
        df = pd.DataFrame([
            {
                "timestamp": "2026-01-01T00:00:00",
                "amount": 10,
                "payer_id": "u1",
                "payee_id": "p1",
                "device_id": "d1",
                "ip_id": "ip1",
                "merchant_id": "m1",
            },
            {
                "timestamp": "2026-01-01T00:04:00",
                "amount": 15,
                "payer_id": "u1",
                "payee_id": "p2",
                "device_id": "d1",
                "ip_id": "ip1",
                "merchant_id": "m1",
            },
        ])

        out = add_offline_window_features(df)

        self.assertEqual(int(out.loc[1, "window_features_user_txn_count_5min"]), 2)
        self.assertEqual(float(out.loc[1, "window_features_user_amount_sum_5min"]), 25.0)
        self.assertEqual(int(out.loc[1, "window_features_user_unique_payee_count_1h"]), 2)


if __name__ == "__main__":
    unittest.main()
