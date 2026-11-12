# Experiment 3: Varied Attacks, Extraction, and Reasoning

**Question:** How much of the difficulty comes from identifying attacks in text, and how much from reasoning over the attack graph?

## Data

[`varied_attacks_50.jsonl`](varied_attacks_50.jsonl) contains 50 newly generated witness puzzles, with their correct directed attack edges and answers. The generation is deterministic (seed 2027).

- **25 `yes`, 25 `no`** answers to Task 1.
- **Five graph structures:** chains, multiple attackers, branching, diamonds, and mixed graphs (10 puzzles each).
- **17 statement templates:** direct and indirect accusations, reversed source/target sentence order, and accusations naming two targets.
- **434 attack edges** across 50 puzzles. Eighteen puzzles also contain a factual statement that is not an attack.

Each case lists the witnesses, their statements, the query witness's claim, and the gold graph. The rules are unchanged: a witness is accepted if no accepted witness attacks them. The graphs have no cycles.

## Method

1. Prompt each model to extract `[{"source": "...", "target": "..."}]` from the puzzle, without asking the model to solve it.
2. Build the attack graph from its predicted edges.
3. Apply the original [graph solver](../argpart_rule_baseline.py) to predict the Task 1 and Task 2 answers.
4. Compare the predicted graph and answers to gold. We report edge precision/recall/F1, exact graph matches, Task 1 macro-F1, and Task 2 exact set matches.

The three models, all within the shared-task 8B cap, are Llama 3.2 3B, Gemma 3 4B, and Llama 3.1 8B. This extraction setting uses 150 calls (three models × 50 puzzles), at temperature 0. The models see only the puzzle text in this setting.

## Comparison on the Same 50 Puzzles

We also tested direct answers from text, and three LLM reasoning settings where the **correct attack graph** is provided: direct answers, explicit step-by-step reasoning, and few-shot chain-of-thought with two worked examples. Each setting evaluates both tasks separately.

Each cell shows **Task 1 accuracy / Task 2 exact match**, in percent:

| Setting and result run | Llama 3B | Gemma 4B | Llama 8B |
| --- | ---: | ---: | ---: |
| [Text → direct LLM](https://github.com/Khalid-AlKhatib/PART/actions/runs/43218016452) | 54 / 0 | 50 / 0 | 42 / 0 |
| [Text → extracted graph → solver](https://github.com/Khalid-AlKhatib/PART/actions/runs/43215144967) | 26 / 18 | **70 / 38** | **62 / 42** |
| [Correct graph → direct LLM](https://github.com/Khalid-AlKhatib/PART/actions/runs/43234031619) | 50 / 2 | 50 / 4 | 40 / 2 |
| [Correct graph → step-by-step LLM](https://github.com/Khalid-AlKhatib/PART/actions/runs/43236627708) | 34 / 0 | 56 / 0 | 36 / 4 |
| [Correct graph → few-shot CoT LLM](https://github.com/Khalid-AlKhatib/PART/actions/runs/43246211923) | 52 / 2 | 54 / 6 | 44 / 6 |
| Correct graph → deterministic solver | 100 / 100 | 100 / 100 | 100 / 100 |

The attack-extraction edge F1 scores were **48.4% (3B), 79.0% (4B), and 67.6% (8B)**. Using the extracted graph with a deterministic solver improved Task 2 over direct LLM answers. Providing the correct graph to an LLM did not solve the problem: even with worked reasoning examples, the best Task 2 exact-match score was 6%.

The [graph extraction code](run_graph_extraction.py), [direct and gold-graph evaluation code](run_direct_varied.py), and [gold-graph CoT workflow](../.github/workflows/gold-graph-cot-eval.yml) are available for reproduction. The linked result runs contain raw responses and summary files.

## Run

```bash
python3 experiments/make_varied_attacks.py
python3 -m unittest discover -s tests -v
python3 experiments/run_graph_extraction.py --dry-run
```

The evaluation workflows and their results are linked in the table above.
