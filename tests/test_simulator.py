from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraudsim.simulator.generate import SimulationConfig, generate


class SimulatorTest(unittest.TestCase):
    def test_generate_configurable_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_dir = root / "dataset"
            ds_dir.mkdir()
            with (ds_dir / "transaction_stream.jsonl").open("w", encoding="utf-8") as f:
                for index in range(20):
                    f.write(json.dumps({
                        "transaction_id": f"base_{index}",
                        "timestamp": "2026-01-01T00:00:00",
                        "amount": 100,
                        "payer_id": f"U_{index}",
                        "payee_id": f"P_{index}",
                        "merchant_id": f"M_{index}",
                        "device_id": f"D_{index}",
                        "ip_id": f"IP_{index}",
                        "payer_country": "CN",
                        "payee_country": "CN",
                    }) + "\n")
            config = SimulationConfig(
                dataset="dataset",
                seed=7,
                rows=100,
                fraud_ratio=0.2,
                fraud_group_count=2,
                fraud_group_size=5,
                shared_device_strength=0.8,
                shared_ip_strength=0.8,
                cross_border_ratio=1.0,
                large_amount_ratio=1.0,
                night_ratio=1.0,
                rate=50,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            out = root / "out"
            summary = generate(config, ds_dir, out)
            self.assertEqual(summary["rows"], 100)
            self.assertEqual(summary["fraud_rows"], 20)
            labels = pd.read_parquet(out / "simulation_labels.parquet")
            self.assertEqual(int(labels["is_fraud"].sum()), 20)
            first = json.loads((out / "transaction_stream.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(first["transaction_id"].startswith("SIM_TXN_"))


if __name__ == "__main__":
    unittest.main()

