"""Tests for the 50-example attack-extraction stress test."""

from collections import Counter
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from make_varied_attacks import generate, SHAPES
from run_graph_extraction import (check_dataset, evaluate_response, gold_edges,
                                  parse_edges, score_model, solve_from_edges)


class TestVariedAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate()

    def test_fixed_deterministic_balanced_dataset(self):
        self.assertEqual(self.rows, generate())
        self.assertEqual(len(self.rows), 50)
        self.assertEqual(Counter(r["gold_binary"] for r in self.rows),
                         {"yes": 25, "no": 25})
        self.assertEqual(Counter(r["shape"] for r in self.rows),
                         {s: 10 for s in SHAPES})
        check_dataset(self.rows)

    def test_text_mentions_all_attack_endpoints(self):
        for row in self.rows:
            prompt = row["prompt"]
            self.assertIn("Statements:\n", prompt)
            self.assertIn("Witnesses:", prompt)
            self.assertNotIn("gold_binary", prompt)
            for source, target in row["gold_edges"]:
                self.assertIn(source, prompt)
                self.assertIn(target, prompt)

    def test_parse_json_extracts_direction_and_deduplicates(self):
        witnesses = ["Alice", "Bob", "Charlie"]
        raw = ('```json\n[{"source":"Witness Bob","target":"Alice"}, '
               '{"source":"Bob","target":"alice"}]\n```')
        self.assertEqual(parse_edges(raw, witnesses), {("Bob", "Alice")})
        self.assertEqual(parse_edges('[["Bob", "Alice"]]', witnesses),
                         {("Bob", "Alice")})
        self.assertEqual(parse_edges('{"attacks":[]}', witnesses), set())
        self.assertIsNone(parse_edges('[{"source":"Joe","target":"Alice"}]',
                                      witnesses))
        self.assertIsNone(parse_edges('No attacks found', witnesses))

    def test_recovered_graph_solves_using_existing_rules(self):
        for row in self.rows:
            recovered = {tuple(x) for x in row["gold_edges"]}
            binary, accepted = solve_from_edges(
                row["witnesses"], recovered, row["claim_speaker"])
            self.assertEqual(binary, row["gold_binary"])
            self.assertEqual(accepted, row["gold_accepted"])

    def test_cyclic_prediction_is_rejected(self):
        row = self.rows[0]
        a, b = row["witnesses"][:2]
        result = evaluate_response(json.dumps([
            {"source": a, "target": b},
            {"source": b, "target": a},
        ]), row)
        self.assertTrue(result["valid_graph_json"])
        self.assertFalse(result["graph_acyclic"])
        self.assertIsNone(result["predicted_binary"])

    def test_oracle_extractions_score_100_percent(self):
        results = []
        for row in self.rows:
            output = json.dumps([{"source": source, "target": target}
                                 for source, target in row["gold_edges"]])
            results.append({"id": row["id"], "extracted": evaluate_response(output, row)})
        scores = score_model(results, self.rows)
        for key in ("edge_micro_f1", "exact_graph_accuracy", "task1_accuracy",
                    "task1_macro_f1", "task2_exact_match", "task2_micro_f1"):
            self.assertEqual(scores[key], 1.0, key)


if __name__ == "__main__":
    unittest.main()
