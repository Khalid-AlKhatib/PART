# PART Rule-Based Baseline

This repository contains a simple **non-LLM baseline** for the [PART shared task](https://argpart.github.io/). It is intended to let the organisers check whether the published puzzle templates can be solved by extracting an attack graph and applying a deterministic reasoning rule.

The baseline uses only Python's standard library. It does **not** train a model, call an LLM, look at the correct labels when predicting, or use hard-coded answers.

## How It Works

For each puzzle, the script:

1. Reads the rule phrase (for example, `not truthful`) and all statements beginning with `Witness`.
2. Adds an attack edge `A -> B` when A states that B `is not truthful` (or uses the corresponding phrase from the puzzle).
3. Treats the one witness making a factual claim as the target for Task 1.
4. For an **acyclic** attack graph, considers a witness believable if *none of their believable attackers* defeats them.
5. Returns `yes`/`no` for Task 1 and the set of believable witnesses for Task 2.

This is a symbolic solver for the current structured template, **not** a general language-understanding system. It deliberately raises an error on a directed cycle instead of silently assuming a unique answer.

## Requirements

- Python 3.9 or newer (no `pip install` needed)

## Quick Test (No Dataset Needed)

Clone the repository and run the included unit tests:

```bash
git clone https://github.com/Khalid-AlKhatib/PART.git
cd PART
python3 -m unittest discover -s tests -v
```

The tests cover a simple attack, a counterattack, multiple attackers, people mentioned only as attack targets, inconsistent paired prompts, and cycles.

## Reproduce the Training-Set Evaluation

1. Download the official **training** JSONL file from the [PART dataset on Hugging Face](https://huggingface.co/datasets/michiel/argpart/tree/main/data).
2. Save it as `data/train.jsonl` (create the `data` directory if needed).
3. Run:

```bash
python3 argpart_rule_baseline.py data/train.jsonl
```

The script prints JSON with the number of parsed examples, parse failures, Task 1 accuracy and macro-F1, Task 2 exact-match accuracy and F1, and examples of any errors.

We obtained the following result on the **3,000 published training instances**:

| Measure | Result |
| --- | ---: |
| Parsed | 3,000 / 3,000 |
| Parse failures | 0 |
| Task 1 accuracy | 100% |
| Task 1 macro-F1 | 1.000 |
| Task 2 exact-match accuracy | 100% |
| Task 2 mean instance F1 | 1.000 |
| Task 2 micro-F1 | 1.000 |

These are **training-set results**, not a held-out test score. The solver might fail on new templates, paraphrases, semantic attacks, cyclic graphs, or altered task rules.

## Generate Predictions for an Unlabeled File

For the official sample-test file (or a similarly formatted JSONL file):

```bash
python3 argpart_rule_baseline.py data/sample_test.jsonl --output predictions.jsonl
```

Each output line contains an `id`, `binary` prediction (`yes` or `no`), and `multi_label` list of accepted witness names:

```json
{"id":"example-1","binary":"yes","multi_label":["Alice","Charlie"]}
```

The input must include `id` and `binary_prompt`; when `multi_label_prompt` is also present, the script checks that both prompts describe the same attack graph. To parse the other prompt instead, add `--prompt-field multi_label_prompt`. Add `--no-pair-check` only when you intentionally want to skip the consistency check.

The prediction file is a **local diagnostic format**. Check the shared task's official submission instructions before submitting outputs to an evaluation server.

## What To Check Next

The 100% training result suggests that the current examples may primarily test recovery of a templated attack graph and deterministic graph reasoning. To measure whether systems can handle language understanding as well, we suggest evaluating this baseline on paraphrases, more varied testimony expressions, ambiguous references, and other examples where keyword extraction cannot reconstruct the graph so easily.

The official dataset is not copied into this repository; please obtain it from its maintainers through the link above.
