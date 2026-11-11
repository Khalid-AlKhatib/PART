# PART: Two Baseline Experiments

We evaluated the [PART shared task](https://argpart.github.io/) using a rule-based solver and ten language models. PART contains witness-testimony puzzles with two tasks: predicting whether a statement is believable (Task 1) and identifying all believable witnesses (Task 2).

## Results at a Glance

| Experiment | Training examples | Task 1 macro-F1 | Task 2 exact match |
| --- | ---: | ---: | ---: |
| Rule-based parser + graph reasoning | 3,000 | **100%** | **100%** |
| Best model at or below 8B: Llama 3.1 8B Instruct | 100 | **50.8%** | **1%** |
| Best of all ten tested: Phi-4 14B | 100 | **56.0%** | **3%** |

The rule-based solver was tested on all 3,000 published training examples; the LLMs used the same balanced 100-example sample. See the [full results for all ten models](experiments/README.md).

## Experiment 1: Rule-Based Baseline

### How It Works

Attack statements follow a predictable pattern:

`Witness X says that Y is [attack expression]`

For example: `Witness Bob says that Alice is lying`.

The wording of the attack (`lying`, `not credible`, etc.) is given in each puzzle's opening rule.

The [baseline program](argpart_rule_baseline.py):

1. Extracts the attack expression and the attacking/attacked witnesses.
2. Constructs the directed attack graph.
3. Applies the stated rule: a witness is believable if no believable witness attacks them.
4. Returns the Task 1 and Task 2 answers.

Example:

```text
Witness Alice says that the train is late
Witness Bob says that Alice is lying
Witness Charlie says that Bob is lying
```

Charlie is believable, so Bob is not. Since Bob is not believable, Alice is believable. The answers are **yes** for Task 1 and **Alice, Charlie** for Task 2.

**Result:** 3,000/3,000 examples parsed, Task 1 macro-F1 = 100%, and Task 2 exact match = 100%. The solver uses text patterns and graph reasoning, without an LLM.

## Experiment 2: Ten Small LLMs

We compared ten models on **100 examples** (50 `yes`, 50 `no`), sending each the original prompts for both tasks: 2,000 attempted calls. We used zero-shot prompting, temperature `0`, and a 1,024-token completion limit.

See the [model list, exact prompt, and full results](experiments/README.md), including valid-answer and error counts.

## Reproduce the Rule-Based Experiment

Download the [training data](https://huggingface.co/datasets/michiel/argpart/tree/main/data) as `data/train.jsonl`, then run:

```bash
python3 argpart_rule_baseline.py data/train.jsonl
```

Run unit tests with `python3 -m unittest discover -s tests -v`.

Both experiments use the **published training set**. The rule-based method assumes the training set's statement patterns and acyclic attack graphs.
