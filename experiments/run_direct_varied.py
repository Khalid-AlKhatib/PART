#!/usr/bin/env python3
"""Direct-answer and gold-graph baselines on 50 varied PART-style puzzles.

The three models, puzzle texts, and API settings match the graph-extraction
experiment. Here the models answer each task directly; no graph is extracted
or passed to the symbolic solver. In gold-graph mode, the models receive the
correct directed edges instead of witness statements. Both tasks get separate
requests. The gold-graph-steps condition asks for explicit witness-by-witness
reasoning before the final answer. The gold-graph-cot condition gives two
worked examples with concise reasoning and correct final answers.

Usage:
  python3 experiments/run_direct_varied.py --dry-run
  python3 experiments/run_direct_varied.py
  python3 experiments/run_direct_varied.py --input-mode gold_graph --output-dir results/gold_graph
  python3 experiments/run_direct_varied.py --input-mode gold_graph_steps --output-dir results/gold_graph_steps
  python3 experiments/run_direct_varied.py --input-mode gold_graph_cot --output-dir results/gold_graph_cot
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

STEP_BY_STEP_SYSTEM_PROMPT = (
    "Solve the witness credibility puzzle by applying its rule step by step. "
    "Briefly show the credibility decision for each witness, starting with "
    "witnesses who have no attackers. Then give the final answer in the exact "
    "format requested at the end of the puzzle."
)

# These demonstrations use different witness names from the evaluated cases.
# Example 1 ends with a believable claimant; Example 2 with an unbelievable one.
COT_EXAMPLES = (
    {
        "witnesses": ["A1", "A2", "A3", "A4"],
        "gold_edges": [["A2", "A3"], ["A3", "A4"]],
        "claim_speaker": "A4",
        "gold_binary": "yes",
        "gold_accepted": ["A1", "A2", "A4"],
        "steps": [
            "A1 and A2 have no attackers, so both are believable.",
            "A3 is attacked by believable A2, so A3 is unbelievable.",
            "A4 is attacked only by unbelievable A3, so A4 is believable.",
        ],
    },
    {
        "witnesses": ["B1", "B2", "B3", "B4", "B5", "B6"],
        "gold_edges": [
            ["B1", "B3"], ["B2", "B3"], ["B5", "B2"],
            ["B3", "B4"], ["B4", "B6"], ["B1", "B6"],
        ],
        "claim_speaker": "B6",
        "gold_binary": "no",
        "gold_accepted": ["B1", "B4", "B5"],
        "steps": [
            "B1 and B5 have no attackers, so both are believable.",
            "B2 is attacked by believable B5, so B2 is unbelievable.",
            "B3 is attacked by believable B1, so B3 is unbelievable.",
            "B4 is attacked only by unbelievable B3, so B4 is believable.",
            "B6 is attacked by believable B1 and B4, so B6 is unbelievable.",
        ],
    },
)


def system_prompt_for(input_mode):
    return (STEP_BY_STEP_SYSTEM_PROMPT if input_mode in
            ("gold_graph_steps", "gold_graph_cot")
            else SYSTEM_PROMPT)


def build_gold_graph_problem(record):
    """Expose the correct input graph without accepted labels or answers."""
    arrows = "\n".join(f"- {source} -> {target}"
                       for source, target in record["gold_edges"])
    return (
        "Witness credibility puzzle. A witness is believable if and only if "
        "no believable witness attacks them. Each directed attack from A to B "
        "means witness A accuses witness B of dishonesty. A believable "
        "attacker defeats its target. The attack graph has no cycles.\n\n"
        f"Witnesses: {', '.join(record['witnesses'])}.\n\n"
        "Directed attacks (source -> target):\n"
        + (arrows if arrows else "(none)")
        + f"\n\nThe factual claim was made by witness {record['claim_speaker']}."
    )


def build_cot_examples(task):
    """Show two correct worked examples of the requested task, before the test case."""
    blocks = []
    for index, example in enumerate(COT_EXAMPLES, 1):
        if task == "binary":
            question = f"Is witness {example['claim_speaker']}'s claim believable?"
            answer = example["gold_binary"]
        elif task == "multi_label":
            question = "Which witnesses are believable?"
            answer = repr(example["gold_accepted"])
        else:
            raise ValueError(f"Unknown task: {task}")
        blocks.append(
            f"Worked example {index}:\n"
            + build_gold_graph_problem(example)
            + "\nQuestion: " + question
            + "\nReasoning:\n- " + "\n- ".join(example["steps"])
            + "\nAnswer: " + answer
        )
    return (
        "Here are two solved examples. Use the same short, witness-by-witness "
        "reasoning for the new puzzle.\n\n"
        + "\n\n".join(blocks)
        + "\n\nNow solve this new puzzle:\n\n"
    )


def build_prompt(record, task, input_mode="text"):
    """Return the task question over either original text or the gold graph."""
    if input_mode == "text":
        puzzle = record["prompt"]
    elif input_mode in ("gold_graph", "gold_graph_steps", "gold_graph_cot"):
        puzzle = build_gold_graph_problem(record)
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")
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
    if input_mode == "gold_graph_steps":
        puzzle += (
            "\n\nWork through the graph in these steps:\n"
            "1. Find witnesses with no incoming attacks. They are believable.\n"
            "2. Find witnesses whose attackers have all been evaluated. "
            "If at least one attacker is believable, mark the witness "
            "unbelievable; otherwise mark the witness believable.\n"
            "3. Repeat until every witness has been evaluated. "
            "Briefly show the decision for each witness.\n"
            "4. End with only the final Answer line in the requested format."
        )
    if input_mode == "gold_graph_cot":
        puzzle = build_cot_examples(task) + puzzle
    return puzzle + "\n\nFinal question: " + question


def evaluate_one(model, record, task, api_key, max_tokens, input_mode="text"):
    start = time.monotonic()
    raw = None
    usage = {}
    error = None
    try:
        response = ask_openrouter(model["id"], build_prompt(record, task, input_mode),
                                  api_key, max_tokens,
                                  system_prompt=system_prompt_for(input_mode))
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
        "input_mode": input_mode,
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
    parser.add_argument("--input-mode",
                        choices=["text", "gold_graph", "gold_graph_steps",
                                 "gold_graph_cot"],
                        default="text")
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
        "input_mode": args.input_mode,
        "sample_ids": [r["id"] for r in cases],
        "system_prompt": system_prompt_for(args.input_mode),
        "user_prompt_example": {
            "id": cases[0]["id"],
            "binary": build_prompt(cases[0], "binary", args.input_mode),
            "multi_label": build_prompt(cases[0], "multi_label", args.input_mode),
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
                                       api_key, args.max_tokens, args.input_mode)
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
