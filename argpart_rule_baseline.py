#!/usr/bin/env python3
"""A transparent, standard-library-only symbolic baseline for PART 2027.

Usage:
    python argpart_rule_baseline.py data/train.jsonl
    python argpart_rule_baseline.py data/test.jsonl --output predictions.jsonl

The solver recovers each directed attack from the literal witness templates,
then computes accepted witnesses recursively (correct for acyclic graphs).
It deliberately uses no language model, learned weights, or external packages.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


WITNESS_RE = re.compile(r"^Witness (.+?) says that (.+)$")
RULE_RE = re.compile(
    r"believable testimony that they are (.+?)\. Now consider the\s+following facts:",
    re.DOTALL,
)


class ParseError(ValueError):
    """A case does not match the published generation templates."""


def parse_case(prompt):
    """Return (speakers, attackers_by_target, queried_speaker, diagnostics)."""
    rule = RULE_RE.search(prompt)
    if not rule:
        raise ParseError("Cannot identify the attack phrase in the rule")
    attack_word = rule.group(1).strip()
    statements = []
    for line in prompt.splitlines():
        if line.startswith("Witness "):
            match = WITNESS_RE.fullmatch(line)
            if not match:
                raise ParseError(f"Unparsable witness statement: {line!r}")
            statements.append((match.group(1), match.group(2)))

    speakers = [name for name, _ in statements]
    if not speakers or len(speakers) != len(set(speakers)):
        raise ParseError("Missing statements or duplicate witness speakers")
    attackers = {name: set() for name in speakers}
    world_speakers = []
    missing_targets = set()
    attack_suffix = " is " + attack_word
    for speaker, statement in statements:
        if statement.endswith(attack_suffix):
            target = statement[: -len(attack_suffix)]
            # Only speakers have actual testimony/argument nodes. A person
            # mentioned solely as a target is not an additional witness node.
            if target in attackers:
                attackers[target].add(speaker)
            else:
                missing_targets.add(target)
        else:
            world_speakers.append(speaker)

    if len(world_speakers) != 1:
        raise ParseError(f"Expected 1 factual claim, found {len(world_speakers)}")
    diagnostics = {
        "witnesses": len(speakers),
        "attack_edges": sum(map(len, attackers.values())),
        "target_only_names": len(missing_targets),
        "attack_phrase": attack_word,
    }
    return speakers, attackers, world_speakers[0], diagnostics


def solve(speakers, attackers):
    """Accept a witness iff none of its accepted attackers defeats it."""
    decisions = {}
    visiting = set()

    def accepted(name):
        if name in decisions:
            return decisions[name]
        if name in visiting:
            raise ParseError("Cyclic attacks: the acyclic solver is not defined")
        visiting.add(name)
        result = not any(accepted(attacker) for attacker in attackers[name])
        visiting.remove(name)
        decisions[name] = result
        return result

    return {name for name in speakers if accepted(name)}


def predict(record, prompt_field="binary_prompt", check_pair=True):
    speakers, attackers, world_speaker, diagnostics = parse_case(
        record[prompt_field]
    )
    if check_pair and "multi_label_prompt" in record and "binary_prompt" in record:
        others, other_attackers, other_world_speaker, _ = parse_case(
            record["multi_label_prompt" if prompt_field == "binary_prompt" else "binary_prompt"]
        )
        if (set(speakers) != set(others) or attackers != other_attackers
                or world_speaker != other_world_speaker):
            raise ParseError("Binary and multi-label prompts describe different graphs")
    believable = solve(speakers, attackers)
    result = {
        "id": record["id"],
        "binary": "yes" if world_speaker in believable else "no",
        "multi_label": sorted(believable),
    }
    return result, diagnostics


def mean(values):
    return sum(values) / len(values) if values else None


def evaluate(results):
    labeled = [r for r in results if r["gold"] is not None]
    if not labeled:
        return None
    tp = {"yes": 0, "no": 0}
    fp = {"yes": 0, "no": 0}
    fn = {"yes": 0, "no": 0}
    binary_ok = 0
    exact_ok = 0
    f1_scores = []
    false_positive = false_negative = 0
    gold_positive = pred_positive = 0
    for row in labeled:
        pred, gold = row["prediction"], row["gold"]
        y, p = gold["binary"], pred["binary"]
        binary_ok += (p == y)
        for label in tp:
            tp[label] += (p == label and y == label)
            fp[label] += (p == label and y != label)
            fn[label] += (p != label and y == label)
        pred_names, gold_names = set(pred["multi_label"]), set(gold["multi_label"])
        exact_ok += (pred_names == gold_names)
        common = len(pred_names & gold_names)
        f1_scores.append(2 * common / (len(pred_names) + len(gold_names))
                         if pred_names or gold_names else 1.0)
        false_positive += len(pred_names - gold_names)
        false_negative += len(gold_names - pred_names)
        gold_positive += len(gold_names)
        pred_positive += len(pred_names)
    label_f1 = {
        label: (2 * tp[label] / (2 * tp[label] + fp[label] + fn[label])
                if 2 * tp[label] + fp[label] + fn[label] else 0.0)
        for label in tp
    }
    return {
        "evaluated": len(labeled),
        "task1_accuracy": binary_ok / len(labeled),
        "task1_macro_f1": mean(list(label_f1.values())),
        "task1_class_f1": label_f1,
        "task2_exact_accuracy": exact_ok / len(labeled),
        "task2_mean_instance_f1": mean(f1_scores),
        "task2_micro_f1": (2 * (pred_positive - false_positive)
                           / (gold_positive + pred_positive)
                           if gold_positive + pred_positive else 1.0),
        "task2_false_positive_names": false_positive,
        "task2_false_negative_names": false_negative,
        "task1_errors": len(labeled) - binary_ok,
        "task2_exact_errors": len(labeled) - exact_ok,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Official PART JSONL file")
    parser.add_argument("--output", type=Path, help="Write predictions as JSONL")
    parser.add_argument("--prompt-field", choices=["binary_prompt", "multi_label_prompt"],
                        default="binary_prompt")
    parser.add_argument("--no-pair-check", action="store_true")
    parser.add_argument("--max-errors", type=int, default=8)
    args = parser.parse_args()

    results, failures = [], []
    diagnostics_totals = Counter()
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                prediction, diagnostics = predict(
                    record, args.prompt_field, not args.no_pair_check
                )
            except ParseError as exc:
                failures.append({"line": line_number, "id": record.get("id"), "error": str(exc)})
                continue
            for key in ("witnesses", "attack_edges", "target_only_names"):
                diagnostics_totals[key] += diagnostics[key]
            if diagnostics["target_only_names"]:
                diagnostics_totals["cases_with_target_only_names"] += 1
            gold = None
            if "binary" in record and "multi_label" in record:
                gold = {"binary": record["binary"], "multi_label": record["multi_label"]}
            results.append({"prediction": prediction, "gold": gold,
                            "diagnostics": diagnostics})

    if args.output:
        with args.output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result["prediction"], ensure_ascii=False) + "\n")
    scores = evaluate(results)
    summary = {
        "total_rows": len(results) + len(failures),
        "parsed": len(results),
        "parse_failures": len(failures),
        "parse_failure_examples": failures[:args.max_errors],
        "diagnostics_totals": dict(diagnostics_totals),
        "scores": scores,
    }
    if scores:
        wrong = [row for row in results if
                 (row["prediction"]["binary"] != row["gold"]["binary"] or
                  set(row["prediction"]["multi_label"]) != set(row["gold"]["multi_label"]))]
        summary["prediction_error_examples"] = [
            {"id": row["prediction"]["id"], "pred": row["prediction"],
             "gold": row["gold"]} for row in wrong[:args.max_errors]
        ]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
