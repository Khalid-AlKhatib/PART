"""Offline checks. These never contact OpenRouter."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_small_llms.py"
SPEC = importlib.util.spec_from_file_location("small_llms", SCRIPT)
small_llms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(small_llms)


class TestSmallLLMExperiment(unittest.TestCase):
    def test_binary_answers(self):
        self.assertEqual(small_llms.parse_binary_answer("Answer: yes"), "yes")
        self.assertEqual(small_llms.parse_binary_answer("Answer: no"), "no")
        self.assertEqual(small_llms.parse_binary_answer("NO"), "no")
        self.assertIsNone(small_llms.parse_binary_answer("probably yes"))
        self.assertIsNone(small_llms.parse_binary_answer("Answer: yes or no"))

    def test_name_lists(self):
        names = ["Alice", "Bob", "Charlie"]
        self.assertEqual(
            small_llms.parse_list_answer("Answer: ['Charlie', 'alice']", names),
            ["Alice", "Charlie"],
        )
        self.assertEqual(small_llms.parse_list_answer("Answer: []", names), [])
        self.assertIsNone(small_llms.parse_list_answer("Answer: ['David']", names))
        self.assertIsNone(small_llms.parse_list_answer("Answer: Alice, Bob", names))

    def test_balanced_sample_is_reproducible(self):
        records = [{"id": str(i), "binary": "yes" if i % 2 else "no",
                    "multi_label": ["Alice"]} for i in range(40)]
        a = small_llms.select_examples(records, 10, 42)
        b = small_llms.select_examples(records, 10, 42)
        self.assertEqual([row["id"] for row in a], [row["id"] for row in b])
        self.assertEqual(sum(row["binary"] == "yes" for row in a), 5)

    def test_metrics_treat_invalid_as_wrong(self):
        rows = [
            {"task": "binary", "gold": "yes", "prediction": "yes", "valid_answer": True,
             "error": None, "usage": {}},
            {"task": "binary", "gold": "no", "prediction": None, "valid_answer": False,
             "error": None, "usage": {}},
            {"task": "multi_label", "gold": ["Alice"], "prediction": ["Alice"],
             "valid_answer": True, "error": None, "usage": {}},
            {"task": "multi_label", "gold": ["Bob"], "prediction": None,
             "valid_answer": False, "error": "Empty response", "usage": {}},
        ]
        scores = small_llms.summarize(rows)
        self.assertEqual(scores["task1_accuracy"], 0.5)
        self.assertEqual(scores["task2_exact_match"], 0.5)
        self.assertEqual(scores["task2_micro_f1"], 2 / 3)
        self.assertEqual(scores["request_errors"], 1)

    def test_cli_dry_run_without_key(self):
        records = [
            {"id": f"x-{i}", "binary": "yes" if i % 2 else "no",
             "multi_label": ["Alice"], "binary_prompt": "unused",
             "multi_label_prompt": "unused"}
            for i in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "train.jsonl"
            data.write_text("\n".join(json.dumps(x) for x in records) + "\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--data", str(data),
                 "--sample-size", "4", "--dry-run", "--output-dir", str(root / "out")],
                capture_output=True, text=True, check=True,
            )
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertTrue(manifest["dry_run"])
            self.assertEqual(len(manifest["sample_ids"]), 4)
            self.assertIn("no API calls", result.stdout)


if __name__ == "__main__":
    unittest.main()
