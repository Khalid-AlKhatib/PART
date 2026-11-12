#!/usr/bin/env python3
"""Generate 50 varied, fully labeled attack-graph puzzles.

This is a small stress test of text-to-graph extraction, not a new official
PART test split. Every direct accusation has a known source and target.

Usage:
    python3 experiments/make_varied_attacks.py \
      --output experiments/varied_attacks_50.jsonl
"""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argpart_rule_baseline import solve


NAMES = (
    "Alice", "Bob", "Charlie", "Dana", "Elias", "Fatima", "Grace",
    "Hassan", "Iris", "Jonah", "Klara", "Leo", "Maya", "Nadia", "Omar",
    "Priya", "Quentin", "Rania", "Sofia", "Tariq", "Uma", "Vera",
    "Wesley", "Yara", "Zoe", "Amir", "Beatrice", "Celia", "Dario",
    "Elena", "Farah", "Gideon", "Hana", "Isaac", "Jasmine", "Kareem",
)

CLAIMS = (
    "the defendant was at the station",
    "the museum was open on Monday",
    "the train arrived before noon",
    "the letter was delivered yesterday",
    "the meeting began at nine",
)
NEUTRAL_FACTS = (
    "the library closes at six",
    "the hearing was held on Thursday",
    "the conference room has two doors",
    "the museum is next to the station",
)

# Each template explicitly asserts the target's dishonesty or unreliability.
# Source and target sometimes appear in reverse order.
SINGLE_ATTACKS = (
    "Witness {s} says that {t} is lying.",
    "{s} claims that {t} fabricated their account.",
    "According to witness {s}, {t}'s testimony cannot be trusted.",
    "Witness {s} calls {t} an unreliable witness.",
    "{t} is accused of giving false testimony by witness {s}.",
    "Witness {s} challenges the honesty of {t}'s account.",
    "In {s}'s testimony, {t} deliberately misled the panel.",
    "It is {s} who alleges that {t}'s story was invented.",
    "Witness {s} says: I believe {t} is not telling the truth.",
    "{s} states that the account offered by {t} is fabricated.",
    "The accusation from witness {s} is that {t} lied to the court.",
    "{t}'s credibility is directly challenged by witness {s}, who says {t} lied.",
)
DOUBLE_ATTACKS = (
    "Witness {s} says that both {a} and {b} fabricated their testimony.",
    "According to {s}, neither {a} nor {b} is telling the truth.",
    "{a} and {b} are accused of lying by witness {s}.",
    "Witness {s} challenges the reliability of two witnesses: {a} and {b}.",
    "In {s}'s view, {a}'s account was invented, and so was {b}'s.",
)

SHAPES = ("chain", "fan_in", "branching", "diamond", "mixed")


def edges_for_shape(shape, n, rng):
    """Return (source_idx, target_idx) pairs. Targets always precede sources."""
    if shape == "chain":
        return {(i, i - 1) for i in range(1, n)}
    if shape == "fan_in":
        edges = {(i, 0) for i in range(1, min(n, 6))}
        for i in range(6, n):
            edges.add((i, rng.randint(1, min(5, i - 1))))
        return edges
    if shape == "branching":
        return {(i, rng.choice([0, rng.randrange(i)])) for i in range(1, n)}
    if shape == "diamond":
        edges = {(1, 0), (2, 0), (3, 1), (3, 2), (4, 3)}
        for i in range(5, n):
            target = rng.randrange(i)
            edges.add((i, target))
            if rng.random() < 0.3:
                edges.add((i, rng.choice([j for j in range(i) if j != target])))
        return edges
    if shape == "mixed":
        edges = set()
        for i in range(1, n):
            target = rng.randrange(i)
            edges.add((i, target))
            if i > 2 and rng.random() < 0.55:
                edges.add((i, rng.choice([j for j in range(i) if j != target])))
        return edges
    raise ValueError(f"Unknown shape: {shape}")


def make_candidate(shape, rng):
    n = rng.randint(7, 11)
    names = rng.sample(NAMES, n)
    edges = edges_for_shape(shape, n, rng)
    # A minority of puzzles include an additional factual statement; this
    # prevents treating every non-query statement as an attack by default.
    neutral_idx = rng.randrange(2, n) if rng.random() < 0.3 else None
    if neutral_idx is not None:
        edges = {(s, t) for s, t in edges if s != neutral_idx}

    graph = {name: set() for name in names}
    for src, tgt in edges:
        graph[names[tgt]].add(names[src])
    accepted = solve(names, graph)
    claim = rng.choice(CLAIMS)
    lines = []
    templates_used = set()
    for i, source in enumerate(names):
        targets = sorted((names[t] for s, t in edges if s == i))
        if i == 0:
            lines.append(f"Witness {source} reports that {claim}.")
        elif not targets:
            lines.append(f"Witness {source} reports that {rng.choice(NEUTRAL_FACTS)}.")
        elif len(targets) == 1:
            template_index = rng.randrange(len(SINGLE_ATTACKS))
            templates_used.add(f"single_{template_index}")
            lines.append(SINGLE_ATTACKS[template_index].format(s=source, t=targets[0]))
        elif len(targets) == 2:
            template_index = rng.randrange(len(DOUBLE_ATTACKS))
            templates_used.add(f"double_{template_index}")
            lines.append(DOUBLE_ATTACKS[template_index].format(
                s=source, a=targets[0], b=targets[1]))
        else:
            raise AssertionError("Only one or two targets per witness are generated")
    rng.shuffle(lines)
    roster = names[:]
    rng.shuffle(roster)
    prompt = (
        "Witness credibility puzzle. A witness is credible if and only if no "
        "credible witness directly accuses them of dishonesty or unreliable "
        "testimony. A direct accusation is an attack from the accuser (source) "
        "to each accused witness (target). Ordinary factual claims are not "
        "attacks. The attack graph has no cycles.\n\n"
        f"Witnesses: {', '.join(roster)}.\n\nStatements:\n"
        + "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
        + f"\n\nThe claim in question is: {claim}."
    )
    expected_edges = sorted([[names[s], names[t]] for s, t in edges])
    return {
        "shape": shape,
        "prompt": prompt,
        "witnesses": sorted(names),
        "claim_speaker": names[0],
        "gold_edges": expected_edges,
        "gold_binary": "yes" if names[0] in accepted else "no",
        "gold_accepted": sorted(accepted),
        "n_witnesses": n,
        "n_attacks": len(expected_edges),
        "has_neutral_witness": neutral_idx is not None,
        "patterns_used": sorted(templates_used),
    }


def generate(count=50, seed=2027):
    if count != 50:
        raise ValueError("This experiment is fixed at 50 instances")
    rng = random.Random(seed)
    data = []
    seen_prompts = set()
    for shape in SHAPES:
        counts = {"yes": 0, "no": 0}
        for _ in range(20000):
            if counts == {"yes": 5, "no": 5}:
                break
            row = make_candidate(shape, rng)
            label = row["gold_binary"]
            if counts[label] == 5 or row["prompt"] in seen_prompts:
                continue
            seen_prompts.add(row["prompt"])
            counts[label] += 1
            row["id"] = f"varied-{len(data)+1:03d}"
            data.append(row)
        if counts != {"yes": 5, "no": 5}:
            raise RuntimeError(f"Cannot balance {shape}: {counts}")
    rng.shuffle(data)
    for idx, row in enumerate(data, 1):
        row["id"] = f"varied-{idx:03d}"
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/varied_attacks_50.jsonl"))
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    data = generate(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in data:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "instances": len(data),
        "yes": sum(row["gold_binary"] == "yes" for row in data),
        "no": sum(row["gold_binary"] == "no" for row in data),
        "shapes": {shape: sum(row["shape"] == shape for row in data)
                   for shape in SHAPES},
        "total_attacks": sum(row["n_attacks"] for row in data),
        "patterns_covered": len(set(p for row in data for p in row["patterns_used"])),
        "neutral_cases": sum(row["has_neutral_witness"] for row in data),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
