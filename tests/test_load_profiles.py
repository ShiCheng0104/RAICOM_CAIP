from __future__ import annotations

import json
import unittest

import pandas as pd

from fraudsim.streaming.load_profiles import write_frame


class FakePipeline:
    def __init__(self, storage: dict[str, str]) -> None:
        self.storage = storage
        self.execute_count = 0

    def set(self, key: str, value: str) -> None:
        self.storage[key] = value

    def execute(self) -> None:
        self.execute_count += 1


class FakeRedis:
    def __init__(self) -> None:
        self.storage: dict[str, str] = {}
        self.last_pipeline: FakePipeline | None = None

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        self.last_pipeline = FakePipeline(self.storage)
        return self.last_pipeline


class LoadProfilesTests(unittest.TestCase):
    def test_write_frame_streams_rows_and_batches_pipeline(self) -> None:
        client = FakeRedis()
        frame = pd.DataFrame(
            [
                {"entity_id": "u1", "score": 0.2, "seen_at": pd.Timestamp("2026-06-12")},
                {"entity_id": None, "score": 0.3, "seen_at": pd.NaT},
                {"entity_id": "u2", "score": None, "seen_at": pd.NaT},
            ]
        )

        written = write_frame(client, frame, "entity_id", "profile:user", batch_size=1)

        self.assertEqual(written, 2)
        self.assertEqual(set(client.storage), {"profile:user:u1", "profile:user:u2"})
        self.assertEqual(json.loads(client.storage["profile:user:u1"])["seen_at"], "2026-06-12T00:00:00")
        self.assertIsNone(json.loads(client.storage["profile:user:u2"])["score"])
        self.assertEqual(client.last_pipeline.execute_count, 3)


if __name__ == "__main__":
    unittest.main()
