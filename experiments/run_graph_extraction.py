#!/usr/bin/env python3
"""Ask small LLMs to extract attack edges, then solve the recovered graphs.

One OpenRouter request per model and puzzle. No gold relations or labels are
included in requests. Responses and scores are saved for inspection.

Usage:
  python3 experiments/run_graph_extraction.py --dry-run
  python3 experiments/run_graph_extraction.py
  python3 experiments/run_graph_extraction.py --score-responses results/graph_extraction/responses.jsonl
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argpart_rule_baseline import ParseError, solve
from run_small_llms import API_URL, check_models_available, http_json
from urllib.error import HTTPError, URLError


SYSTEM_PROMPT = (
    "You extract direct credibility attacks from witness statements. "
    "An attack has a source (the witness making the accusation) and a target "
    "(the witness accused of lying, fabricating, misleading, or being unreliable). "
    "One statement may attack two witnesses. Ordinary factual statements "
    "are not attacks. Do not decide who is credible; only extract attacks. "
    'Return only a JSON array of objects: [{"source":"Name","target":"Name"}]. '
    "Use the witness names exactly as written. Return [] if there are no attacks."
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_edges(raw, witnesses):
    """Normalize JSON array outputs; None means no usable graph was returned."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    clean = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
    if fence:
        clean = fence.group(1).strip()
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        # Some models include a brief preface. Accept a single embedded JSON
        # array if its full content parses, while retaining the raw response.
        start = clean.find("[")
        if start < 0:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(clean[start:])
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        value = value.get("attacks", value.get("edges"))
    if not isinstance(value, list):
        return None
    canonical = {name.casefold(): name for name in witnesses}
    result = set()
    for pair in value:
        if isinstance(pair, dict):
            src, tgt = pair.get("source"), pair.get("target")
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            src, tgt = pair
        else:
            return None
        if not isinstance(src, str) or not isinstance(tgt, str):
            return None
        src = canonical.get(re.sub(r"^witness\s+", "", src.strip(), flags=re.I).casefold())
        tgt = canonical.get(re.sub(r"^witness\s+", "", tgt.strip(), flags=re.I).casefold())
        if src is None or tgt is None:
            return None
        result.add((src, tgt))
    return result


def gold_edges(row):
    return {tuple(pair) for pair in row["gold_edges"]}


def check_dataset(rows):
    """Check that every puzzle's gold labels agree with the graph solver."""
    if len(rows) != 50 or len({r["id"] for r in rows}) != 50:
        raise ValueError("Expected 50 uniquely identified puzzles")
    if Counter(r["gold_binary"] for r in rows) != {"yes": 25, "no": 25}:
        raise ValueError("Expected 25 yes and 25 no labels")
    for row in rows:
        witnesses = row["witnesses"]
        if len(set(witnesses)) != len(witnesses):
            raise ValueError(f"Duplicate witness names in {row['id']}")
        edges = gold_edges(row)
        if any(s not in witnesses or t not in witnesses for s, t in edges):
            raise ValueError(f"Unknown witness in {row['id']}")
        speakers = {name: set() for name in witnesses}
        for source, target in edges:
            speakers[target].add(source)
        accepted = solve(witnesses, speakers)
        if sorted(accepted) != row["gold_accepted"]:
            raise ValueError(f"Incorrect accepted set in {row['id']}")
        if ("yes" if row["claim_speaker"] in accepted else "no") != row["gold_binary"]:
            raise ValueError(f"Incorrect binary label in {row['id']}")


def solve_from_edges(witnesses, edges, claim_speaker):
    attackers = {name: set() for name in witnesses}
    for source, target in edges:
        attackers[target].add(source)
    # Explicitly reject any predicted cycle, including cycles in branches
    # that might be skipped by a short-circuiting recursive truth evaluation.
    pending = set()
    finished = set()

    def visit(name):
        if name in pending:
            raise ParseError("Predicted attack graph contains a cycle")
        if name in finished:
            return
        pending.add(name)
        for attacker in attackers[name]:
            visit(attacker)
        pending.remove(name)
        finished.add(name)

    for name in witnesses:
        visit(name)
    accepted = solve(witnesses, attackers)
    return "yes" if claim_speaker in accepted else "no", sorted(accepted)


def evaluate_response(raw, row):
    predicted_edges = parse_edges(raw, row["witnesses"])
    valid = predicted_edges is not None
    predicted_binary = None
    predicted_accepted = None
    reasoning_error = None
    if valid:
        try:
            predicted_binary, predicted_accepted = solve_from_edges(
                row["witnesses"], predicted_edges, row["claim_speaker"])
        except (ParseError, KeyError) as exc:
            reasoning_error = str(exc)
    return {
        "valid_graph_json": valid,
        "graph_acyclic": valid and reasoning_error is None,
        "reasoning_error": reasoning_error,
        "predicted_edges": sorted([list(pair) for pair in predicted_edges]) if valid else None,
        "predicted_binary": predicted_binary,
        "predicted_accepted": predicted_accepted,
    }


def score_model(responses, cases):
    by_id = {row["id"]: row for row in cases}
    if len(responses) != len(cases) or len({r["id"] for r in responses}) != len(cases):
        raise ValueError("Expected one response per example")
    tp = fp = fn = graph_exact = binary_ok = multi_exact = 0
    accepted_tp = accepted_fp = accepted_fn = 0
    yes_tp = no_tp = yes_fp = no_fp = yes_fn = no_fn = 0
    valid = acyclic = errors = 0
    latencies = []
    costs = []
    by_shape = {}
    for response in responses:
        gold = by_id[response["id"]]
        pred = response["extracted"]
        valid += int(pred["valid_graph_json"])
        acyclic += int(pred["graph_acyclic"])
        errors += int(bool(response.get("error")))
        if isinstance(response.get("seconds"), (float, int)):
            latencies.append(response["seconds"])
        if isinstance(response.get("usage"), dict):
            cost = response["usage"].get("cost")
            if isinstance(cost, (int, float)):
                costs.append(float(cost))

        true = gold_edges(gold)
        guessed = ({tuple(p) for p in pred["predicted_edges"]}
                   if pred["predicted_edges"] is not None else set())
        tp += len(true & guessed)
        fp += len(guessed - true)
        fn += len(true - guessed)
        exact = pred["graph_acyclic"] and true == guessed
        graph_exact += int(exact)
        correct_binary = pred["predicted_binary"] == gold["gold_binary"]
        binary_ok += int(correct_binary)
        correct_multi = pred["predicted_accepted"] == gold["gold_accepted"]
        multi_exact += int(correct_multi)
        actual_acc = set(gold["gold_accepted"])
        guessed_acc = set(pred["predicted_accepted"] or [])
        accepted_tp += len(actual_acc & guessed_acc)
        accepted_fp += len(guessed_acc - actual_acc)
        accepted_fn += len(actual_acc - guessed_acc)
        for label in ("yes", "no"):
            num_tp = gold["gold_binary"] == label and pred["predicted_binary"] == label
            num_fp = gold["gold_binary"] != label and pred["predicted_binary"] == label
            num_fn = gold["gold_binary"] == label and pred["predicted_binary"] != label
            if label == "yes":
                yes_tp += num_tp
                yes_fp += num_fp
                yes_fn += num_fn
            else:
                no_tp += num_tp
                no_fp += num_fp
                no_fn += num_fn
        by_shape.setdefault(gold["shape"], [0, 0])
        by_shape[gold["shape"]][0] += int(exact)
        by_shape[gold["shape"]][1] += 1

    f1 = lambda x, y, z: 2*x/(2*x+y+z) if (2*x+y+z) else 0.0
    total = len(cases)
    return {
        "n": total,
        "valid_graph_json": valid,
        "acyclic_extractions": acyclic,
        "api_or_empty_errors": errors,
        "edge_precision": tp / (tp + fp) if tp + fp else 0.0,
        "edge_recall": tp / (tp + fn) if tp + fn else 0.0,
        "edge_micro_f1": f1(tp, fp, fn),
        "edge_true_positives": tp,
        "edge_false_positives": fp,
        "edge_false_negatives": fn,
        "exact_graph_accuracy": graph_exact / total,
        "task1_accuracy": binary_ok / total,
        "task1_macro_f1": (f1(yes_tp, yes_fp, yes_fn) + f1(no_tp, no_fp, no_fn))/2,
        "task2_exact_match": multi_exact / total,
        "task2_micro_f1": f1(accepted_tp, accepted_fp, accepted_fn),
        "exact_graph_by_shape": {k: v[0]/v[1] for k, v in sorted(by_shape.items())},
        "average_seconds": round(sum(latencies)/len(latencies), 3) if latencies else None,
        "reported_cost_usd": round(sum(costs), 6) if costs else None,
    }


def ask(model, prompt, api_key, max_tokens):
    payload = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        try:
            return http_json(API_URL, body=payload, key=api_key)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            if attempt == 2:
                raise RuntimeError("OpenRouter timeout or connection error") from exc
        time.sleep(2**attempt)
    raise RuntimeError("Retries exhausted")


def evaluate_one(model, row, api_key, max_tokens):
    start = time.monotonic()
    raw = None
    usage = {}
    error = None
    try:
        response = ask(model, row["prompt"], api_key, max_tokens)
        raw = response["choices"][0]["message"].get("content") or ""
        usage = response.get("usage") or {}
        if not isinstance(raw, str) or not raw:
            error = "Empty response"
    except (RuntimeError, KeyError, IndexError, TypeError, ValueError) as exc:
        error = str(exc)
    return {
        "model": model["id"],
        "id": row["id"],
        "raw_response": raw,
        "extracted": evaluate_response(raw, row),
        "error": error,
        "seconds": round(time.monotonic() - start, 3),
        "usage": usage,
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
                        default=Path("results/graph_extraction"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-responses", type=Path,
                        help="Re-evaluate saved responses without sending API calls")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    cases = read_jsonl(args.data)
    models = json.loads(args.models.read_text(encoding="utf-8"))
    check_dataset(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "models": models,
        "data": str(args.data),
        "sample_ids": [row["id"] for row in cases],
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "n_calls": len(cases) * len(models),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Validated {len(cases)} cases, {len(models)} models; "
              f"{len(cases)*len(models)} prospective calls; none sent.")
        return

    if args.score_responses:
        responses = read_jsonl(args.score_responses)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            parser.error("OPENROUTER_API_KEY is required for live evaluation")
        check_models_available(models)
        futures = []
        responses = []
        with (args.output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for model in models:
                    for row in cases:
                        futures.append(pool.submit(
                            evaluate_one, model, row, api_key, args.max_tokens))
                for future in as_completed(futures):
                    result = future.result()
                    responses.append(result)
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    if len(responses) % 10 == 0:
                        print(f"Completed {len(responses)}/{len(futures)} calls", flush=True)

    summaries = {}
    for model in models:
        subset = [r for r in responses if r["model"] == model["id"]]
        summaries[model["id"]] = score_model(subset, cases)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
