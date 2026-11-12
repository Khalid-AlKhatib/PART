# PART: Baselines and Graph Reasoning Experiments

We evaluated the [PART shared task](https://argpart.github.io/) with a rule-based solver and language models, then studied attack extraction and graph reasoning separately. PART contains witness-testimony puzzles with two tasks: predicting whether a statement is believable (Task 1) and identifying all believable witnesses (Task 2).

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

## Experiment 3: Varied Attacks and Graph Reasoning

We generated [50 more varied witness puzzles](experiments/varied_attacks.md) to test two challenges separately: identifying attack relations in text and reasoning over the resulting graph. The puzzles contain 17 attack-expression templates and five graph structures, with 25 `yes` and 25 `no` answers. All LLM settings used the same three small models and 50 cases.

**Results (%):** Each cell shows **Task 1 accuracy / Task 2 exact match**.

| Setting | Llama 3B | Gemma 4B | Llama 8B |
| --- | ---: | ---: | ---: |
| Original text → LLM answers | 54 / 0 | 50 / 0 | 42 / 0 |
| Original text → LLM extracts attacks → graph solver | 26 / 18 | **70 / 38** | **62 / 42** |
| Correct graph → LLM answers | 50 / 2 | 50 / 4 | 40 / 2 |
| Correct graph → LLM, step-by-step | 34 / 0 | 56 / 0 | 36 / 4 |
| Correct graph → LLM, two worked CoT examples | 52 / 2 | 54 / 6 | 44 / 6 |
| Correct graph → deterministic solver | 100 / 100 | 100 / 100 | 100 / 100 |

**Main finding:** Even with the correct attack graph, small LLMs struggle to apply the credibility rule; explicit steps and worked examples help little. Extracting attacks and then using a graph solver performs much better for the 4B and 8B models, though imperfect relation extraction still limits performance. See the [experiment details, prompts, and result runs](experiments/varied_attacks.md).

## Reproduce the Rule-Based Experiment

Download the [training data](https://huggingface.co/datasets/michiel/argpart/tree/main/data) as `data/train.jsonl`, then run:

```bash
python3 argpart_rule_baseline.py data/train.jsonl
```

Run unit tests with `python3 -m unittest discover -s tests -v`.

Experiments 1–2 use the **published training set**; Experiment 3 uses newly generated puzzles. The rule-based method assumes acyclic attack graphs.
