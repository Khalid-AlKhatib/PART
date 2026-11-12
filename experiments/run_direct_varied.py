#!/usr/bin/env python3
"""Direct-answer baseline on the same 50 varied PART-style puzzles.

The three models, puzzle texts, and API settings match the graph-extraction
experiment. Here the models answer each task directly; no graph is extracted
or passed to the symbolic solver. Task 1 and Task 2 get separate requests.

Usage:
  python3 experiments/run_direct_varied.py --dry-run
  python3 experiments/run_direct_varied.py
  python3 experiments/run_direct_varied.py --score-responses results/direct_varied/responses.jsonl
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_graph_extraction import check_dataset, read_jsonl
from run_small_llms import (SYSTEM_PROMPT, ask_openrouter,
                            check_models_available, parse_binary_answer,
                            parse_list_answer, summarize)


TASKS = ("binary", "multi_label")


def build_prompt(record, task):
    """Return the same puzzle and a task-specific question; never include gold."""
    if task == "binary":
        # Naming the queried speaker is conservative toward direct prompting:
        # the graph-based evaluator also knows this identity from metadata.
        question = (
            f"Is witness {record['claim_speaker']}'s claim believable, "
            "according to the witness credibility rule?\n"
            'End your answer with exactly "Answer: yes" or "Answer: no".'
        )
    elif task == "multi_label":
        question = (
            "Which of the listed witnesses are believable according to the rule?\n"
            "End your answer with a list of names, for example: "
            "Answer: ['Alice', 'Bob']. Use Answer: [] if none are believable."
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return record["prompt"] + "\n\nFinal question: " + question


def evaluate_one(model, record, task, api_key, max_tokens):
    start = time.monotonic()
    raw = None
    usage = {}
    error = None
    try:
        response = ask_openrouter(model["id"], build_prompt(record, task),
                                  api_key, max_tokens)
        content = response["choices"][0]["message"].get("content")
        raw = content if isinstance(content, str) else ""
        usage = response.get("usage") or {}
        if not raw:
            error = "Empty response"
    except (RuntimeError, KeyError, IndexError, TypeError, ValueError) as exc:
        error = str(exc)
    prediction = None
    if isinstance(raw, str):
        if task == "binary":
            prediction = parse_binary_answer(raw)
        else:
            prediction = parse_list_answer(raw, record["witnesses"])
    return {
        "model": model["id"],
        "id": record["id"],
        "task": task,
        "gold": record["gold_binary" if task == "binary" else "gold_accepted"],
        "prediction": prediction,
        "valid_answer": prediction is not None,
        "raw_response": raw,
        "error": error,
        "usage": usage,
        "seconds": round(time.monotonic() - start, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("experiments/varied_attacks_50.jsonl"))
    parser.add_argument("--models", type=Path,
                        default=Path("experiments/graph_models.json"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/direct_varied"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-responses", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    cases = read_jsonl(args.data)
    check_dataset(cases)
    models = json.loads(args.models.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "models": models,
        "sample_ids": [r["id"] for r in cases],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_example": {
            "id": cases[0]["id"],
            "binary": build_prompt(cases[0], "binary"),
            "multi_label": build_prompt(cases[0], "multi_label"),
        },
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "n_calls": len(cases) * len(models) * len(TASKS),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Validated {len(cases)} cases, {len(models)} models, "
              f"{len(TASKS)} tasks; {manifest['n_calls']} prospective calls; none sent.")
        return

    if args.score_responses:
        responses = read_jsonl(args.score_responses)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            parser.error("OPENROUTER_API_KEY is required for live evaluation")
        check_models_available(models)
        responses = []
        with (args.output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate_one, model, record, task,
                                       api_key, args.max_tokens)
                           for model in models for record in cases for task in TASKS]
                for future in as_completed(futures):
                    row = future.result()
                    responses.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if len(responses) % 20 == 0:
                        print(f"Completed {len(responses)}/{len(futures)} calls", flush=True)

    summaries = {}
    expected = {(record["id"], task) for record in cases for task in TASKS}
    for model in models:
        subset = [r for r in responses if r["model"] == model["id"]]
        actual = [(r["id"], r["task"]) for r in subset]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"Missing or duplicated responses for {model['id']}")
        summaries[model["id"]] = summarize(subset)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
