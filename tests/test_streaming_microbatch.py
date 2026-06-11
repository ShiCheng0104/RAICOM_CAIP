import json
import unittest

from fraudsim.streaming.flink_job import MicroBatchWindowScorer


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse({
            "results": [
                {
                    "transaction_id": record.get("transaction_id"),
                    "risk_score": 0.91,
                    "risk_level": "high",
                    "decision": "reject",
                }
                for record in json["records"]
            ]
        })


class MicroBatchWindowScorerTest(unittest.TestCase):
    def test_configures_api_key_header(self):
        scorer = MicroBatchWindowScorer(
            api_url="http://model-api:8000",
            redis_url="redis://redis:6379/0",
            api_key="demo-secret",
        )
        self.assertEqual(scorer.http_session.headers["X-API-Key"], "demo-secret")

    def test_flushes_records_together(self):
        scorer = MicroBatchWindowScorer(
            api_url="http://model-api:8000",
            redis_url="redis://redis:6379/0",
            batch_size=2,
            linger_ms=0,
        )
        scorer.http_session = _FakeSession()

        event_1 = {"transaction_id": "tx1", "timestamp": "2026-01-01T00:00:00", "amount": 10}
        event_2 = {"transaction_id": "tx2", "timestamp": "2026-01-01T00:00:01", "amount": 20}

        self.assertEqual(scorer.flat_map(json.dumps(event_1)), [])
        outputs = scorer.flat_map(json.dumps(event_2))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(scorer.http_session.calls), 1)
        self.assertEqual(len(scorer.http_session.calls[0]["json"]["records"]), 2)
        self.assertEqual([json.loads(row)["transaction_id"] for row in outputs], ["tx1", "tx2"])

    def test_flush_marker_emits_tail_batch(self):
        scorer = MicroBatchWindowScorer(
            api_url="http://model-api:8000",
            redis_url="redis://redis:6379/0",
            batch_size=10,
            linger_ms=0,
        )
        scorer.http_session = _FakeSession()

        event = {"transaction_id": "tail_tx", "timestamp": "2026-01-01T00:00:00", "amount": 10}
        self.assertEqual(scorer.flat_map(json.dumps(event)), [])

        outputs = scorer.flat_map(json.dumps({"__fraudsim_flush": True}))

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(scorer.http_session.calls), 1)
        self.assertEqual(json.loads(outputs[0])["transaction_id"], "tail_tx")
