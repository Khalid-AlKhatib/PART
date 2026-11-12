# Experiment 3: Extracting Attacks Before Graph Reasoning

**Question:** Can small LLMs identify attack relations when the language and graph patterns are more varied? Once the relations are extracted, we apply the same deterministic graph reasoning used in Experiment 1.

## Data

[`varied_attacks_50.jsonl`](varied_attacks_50.jsonl) contains 50 newly generated witness puzzles, with their correct directed attack edges and answers. The generation is deterministic (seed 2027).

- **25 `yes`, 25 `no`** answers to Task 1.
- **Five graph structures:** chains, multiple attackers, branching, diamonds, and mixed graphs (10 puzzles each).
- **17 statement templates:** direct and indirect accusations, reversed source/target sentence order, and accusations naming two targets.
- **434 attack edges** across 50 puzzles. Eighteen puzzles also contain a factual statement that is not an attack.

Each case lists the witnesses, their statements, the query witness's claim, and the gold graph. The LLM only receives the puzzle text, not the gold information. The rules are unchanged: a witness is accepted if no accepted witness attacks them. The graphs have no cycles.

## Method

1. Prompt each model to extract `[{"source": "...", "target": "..."}]` from the puzzle, without asking the model to solve it.
2. Build the attack graph from its predicted edges.
3. Apply the original [graph solver](../argpart_rule_baseline.py) to predict the Task 1 and Task 2 answers.
4. Compare the predicted graph and answers to gold. We report edge precision/recall/F1, exact graph matches, Task 1 macro-F1, and Task 2 exact set matches.

The three models, all within the shared-task 8B cap, are Llama 3.2 3B, Gemma 3 4B, and Llama 3.1 8B. There are **150 calls** in total (three models × 50 puzzles), at temperature 0.

## Run

```bash
python3 experiments/make_varied_attacks.py
python3 -m unittest discover -s tests -v
python3 experiments/run_graph_extraction.py --dry-run
```

To evaluate the LLMs, run the [Varied Attack Graph Extraction GitHub Action](../.github/workflows/graph-extraction-eval.yml) with the `RUN` confirmation, using the configured `OPENROUTER_API_KEY` secret. Its artifact contains raw responses, extracted edges, predictions, and per-model results. The evaluator can also score saved responses without API calls.
