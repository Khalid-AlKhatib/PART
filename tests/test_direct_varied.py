"""Checks for the direct-answer comparison on the varied attack dataset."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from make_varied_attacks import generate
from run_direct_varied import build_prompt, build_gold_graph_problem
from run_small_llms import parse_binary_answer, parse_list_answer


class TestDirectVaried(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate()

    def test_same_puzzles_and_no_gold_labels(self):
        for row in self.records:
            for task in ("binary", "multi_label"):
                prompt = build_prompt(row, task)
                self.assertTrue(prompt.startswith(row["prompt"]))
                self.assertNotIn("gold_binary", prompt)
                self.assertNotIn("gold_edges", prompt)
                self.assertNotIn("gold_accepted", prompt)
                self.assertIn("Final question:", prompt)

    def test_direct_answers_parse_with_existing_scoring(self):
        record = self.records[0]
        binary = build_prompt(record, "binary")
        multi = build_prompt(record, "multi_label")
        self.assertIn(record["claim_speaker"], binary)
        self.assertIn("Which of the listed witnesses", multi)
        self.assertEqual(parse_binary_answer("Answer: yes"), "yes")
        self.assertEqual(parse_list_answer("Answer: ['Alice', 'Bob']",
                                           ["Alice", "Bob"]), ["Alice", "Bob"])

    def test_gold_graph_prompts_contain_edges_without_labels(self):
        for record in self.records:
            puzzle = build_gold_graph_problem(record)
            self.assertNotIn("Statements:", puzzle)
            self.assertIn("Directed attacks (source -> target):", puzzle)
            self.assertIn(record["claim_speaker"], puzzle)
            for source, target in record["gold_edges"]:
                self.assertIn(f"- {source} -> {target}", puzzle)
            for task in ("binary", "multi_label"):
                prompt = build_prompt(record, task, "gold_graph")
                self.assertTrue(prompt.startswith(puzzle))
                self.assertNotIn("gold_binary", prompt)
                self.assertNotIn("gold_accepted", prompt)
                self.assertIn("Final question:", prompt)


if __name__ == "__main__":
    unittest.main()
