"""Small standalone checks for the rule-based PART solver."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from argpart_rule_baseline import ParseError, evaluate, predict


SCRIPT = Path(__file__).resolve().parents[1] / "argpart_rule_baseline.py"


def make_prompt(statements, phrase="lying"):
    return (
        "The following is a reasoning puzzle. Witnesses should always be believed\n"
        f"unless there is believable testimony that they are {phrase}. "
        "Now consider the\nfollowing facts:\n\n"
        + "\n".join(f"Witness {person} says that {claim}" for person, claim in statements)
        + "\n\nFinal question: can we believe the claim?"
    )


def make_case(statements, phrase="lying", case_id="example-1"):
    prompt = make_prompt(statements, phrase)
    return {"id": case_id, "binary_prompt": prompt, "multi_label_prompt": prompt}


class TestRuleBasedBaseline(unittest.TestCase):
    def check_answer(self, statements, binary, names, phrase="lying"):
        prediction, _ = predict(make_case(statements, phrase))
        self.assertEqual(prediction["binary"], binary)
        self.assertEqual(prediction["multi_label"], sorted(names))

    def test_no_attacks(self):
        self.check_answer([("Alice", "the defendant is innocent")], "yes", ["Alice"])

    def test_single_attack(self):
        self.check_answer(
            [("Alice", "the defendant is innocent"), ("Bob", "Alice is lying")],
            "no", ["Bob"],
        )

    def test_counterattack(self):
        self.check_answer(
            [("Alice", "the defendant is innocent"), ("Bob", "Alice is lying"),
             ("Charlie", "Bob is lying")],
            "yes", ["Alice", "Charlie"],
        )

    def test_two_independent_attackers(self):
        self.check_answer(
            [("Alice", "the defendant is innocent"), ("Bob", "Alice is lying"),
             ("Charlie", "Alice is lying"), ("Dana", "Bob is lying")],
            "no", ["Charlie", "Dana"],
        )

    def test_mentioned_target_without_statement(self):
        self.check_answer(
            [("Alice", "the defendant is innocent"), ("Bob", "Carol is lying")],
            "yes", ["Alice", "Bob"],
        )

    def test_attack_phrase_read_from_prompt(self):
        self.check_answer(
            [("Alice", "the defendant is innocent"),
             ("Bob", "Alice is not truthful")],
            "no", ["Bob"], phrase="not truthful",
        )

    def test_paired_prompt_must_match(self):
        record = make_case([("Alice", "a fact"), ("Bob", "Alice is lying")])
        record["multi_label_prompt"] = make_prompt(
            [("Alice", "a fact"), ("Bob", "Carol is lying")]
        )
        with self.assertRaises(ParseError):
            predict(record)

    def test_cycles_rejected(self):
        record = make_case(
            [("Alice", "a fact"), ("Bob", "Charlie is lying"),
             ("Charlie", "Bob is lying")]
        )
        with self.assertRaises(ParseError):
            predict(record)

    def test_cli_evaluation_and_output(self):
        record = make_case(
            [("Alice", "a fact"), ("Bob", "Alice is lying"),
             ("Charlie", "Bob is lying")]
        )
        record.update(binary="yes", multi_label=["Alice", "Charlie"])
        negative = make_case(
            [("Alice", "a fact"), ("Bob", "Alice is lying")],
            case_id="example-2",
        )
        negative.update(binary="no", multi_label=["Bob"])
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "train.jsonl"
            output_path = Path(tmp) / "predictions.jsonl"
            input_path.write_text(
                json.dumps(record) + "\n" + json.dumps(negative) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(input_path),
                 "--output", str(output_path)],
                check=True, capture_output=True, text=True,
            )
            summary = json.loads(result.stdout)
            self.assertEqual(summary["scores"]["task1_macro_f1"], 1.0)
            self.assertEqual(summary["scores"]["task2_exact_accuracy"], 1.0)
            predictions = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(predictions, [
                {"id": "example-1", "binary": "yes",
                 "multi_label": ["Alice", "Charlie"]},
                {"id": "example-2", "binary": "no", "multi_label": ["Bob"]},
            ])


if __name__ == "__main__":
    unittest.main()
