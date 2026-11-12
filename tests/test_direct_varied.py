"""Checks for the direct-answer comparison on the varied attack dataset."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from make_varied_attacks import generate
from run_direct_varied import build_prompt
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


if __name__ == "__main__":
    unittest.main()
