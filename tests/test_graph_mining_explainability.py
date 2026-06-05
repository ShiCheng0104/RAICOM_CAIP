import unittest

from fraudsim.graph_mining import build_group_subgraph, group_explanations


class GraphMiningExplainabilityTest(unittest.TestCase):
    def test_group_explanations_include_evidence_codes(self):
        group = {
            "fraud_group_mining_id": "GM_TEST",
            "graph_mining_group_risk_score": 0.82,
            "user_count": 6,
            "resource_count": 3,
            "fraud_seed_count": 4,
            "shared_device_count": 2,
            "shared_ip_count": 1,
            "shared_merchant_count": 0,
            "shared_payee_count": 1,
            "scenario_count": 2,
            "top_scenarios": '{"mule_account": 3}',
        }
        evidence = [
            {"entity_id": "D_1", "entity_type": "device", "evidence_kind": "shared_device", "shared_user_count": 3, "fraud_seed_count": 2, "resource_risk_score": 0.7},
            {"entity_id": "IP_1", "entity_type": "ip", "evidence_kind": "shared_ip", "shared_user_count": 4, "fraud_seed_count": 1, "resource_risk_score": 0.6},
        ]

        explanations = group_explanations(group, evidence)

        self.assertEqual(explanations["risk_level"], "high")
        self.assertIn("high_confidence_ring", explanations["explanation_codes"])
        self.assertIn("shared_device_ring", explanations["explanation_codes"])
        self.assertIn("fund_transfer_chain", explanations["explanation_codes"])
        self.assertEqual(explanations["evidence_summary"]["by_kind"]["shared_device"], 1)

    def test_subgraph_contains_group_risk_members_and_resources(self):
        group = {
            "fraud_group_mining_id": "GM_TEST",
            "graph_mining_group_risk_score": 0.66,
            "sample_users": '["u1", "u2"]',
            "sample_resources": '["D_1", "IP_1"]',
        }
        evidence = [
            {"entity_id": "D_1", "entity_type": "device", "evidence_kind": "shared_device", "shared_user_count": 2, "resource_risk_score": 0.5},
            {"entity_id": "IP_1", "entity_type": "ip", "evidence_kind": "shared_ip", "shared_user_count": 3, "resource_risk_score": 0.6},
        ]

        subgraph = build_group_subgraph(group, evidence)

        node_ids = {node["id"] for node in subgraph["nodes"]}
        edge_labels = {edge["label"] for edge in subgraph["edges"]}
        self.assertTrue({"GM_TEST", "GM_TEST:risk", "u1", "u2", "D_1", "IP_1"}.issubset(node_ids))
        self.assertTrue({"member", "shared_device", "shared_ip", "risk_score"}.issubset(edge_labels))


if __name__ == "__main__":
    unittest.main()
