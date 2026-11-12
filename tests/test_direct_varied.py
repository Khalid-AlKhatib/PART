"""Checks for the direct-answer comparison on the varied attack dataset."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from make_varied_attacks import generate
from run_direct_varied import (STEP_BY_STEP_SYSTEM_PROMPT, build_prompt,
                               build_gold_graph_problem, evaluate_one,
                               system_prompt_for)
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

    def test_step_by_step_prompts_require_reasoning_without_labels(self):
        for record in self.records:
            for task in ("binary", "multi_label"):
                prompt = build_prompt(record, task, "gold_graph_steps")
                original = build_prompt(record, task, "gold_graph")
                self.assertIn("no incoming attacks", prompt)
                self.assertIn("Repeat until every witness", prompt)
                self.assertIn("Briefly show the decision", prompt)
                self.assertTrue(prompt.startswith(build_gold_graph_problem(record)))
                self.assertNotEqual(prompt, original)
                self.assertNotIn("gold_accepted", prompt)
                self.assertNotIn("gold_binary", prompt)

    def test_step_by_step_uses_explanation_system_prompt(self):
        self.assertEqual(system_prompt_for("gold_graph_steps"),
                         STEP_BY_STEP_SYSTEM_PROMPT)
        self.assertIn("step by step", STEP_BY_STEP_SYSTEM_PROMPT)
        self.assertIn("Do not include an explanation", system_prompt_for("gold_graph"))
        with patch("run_direct_varied.ask_openrouter") as ask:
            ask.return_value = {"choices": [{"message": {"content": "Answer: no"}}]}
            row = evaluate_one({"id": "test-model"}, self.records[0], "binary",
                               "dummy-key", 1536, "gold_graph_steps")
            self.assertEqual(row["input_mode"], "gold_graph_steps")
            self.assertEqual(ask.call_args.kwargs["system_prompt"],
                             STEP_BY_STEP_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
