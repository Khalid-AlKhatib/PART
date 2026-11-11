# PART: Rule-Based and Small-LLM Experiments

We ran two experiments on the [PART shared task](https://argpart.github.io/): a **rule-based baseline** and a **comparison of ten small language models**. Both address the task's witness-testimony puzzles: Task 1 predicts whether a statement can be believed; Task 2 identifies which witnesses can be believed.

## Results at a Glance

| Experiment | Training examples | Task 1 macro-F1 | Task 2 exact match |
| --- | ---: | ---: | ---: |
| Rule-based parser + graph reasoning | 3,000 | **100%** | **100%** |
| Best model at or below 8B: Llama 3.1 8B Instruct | 100 | **50.8%** | **1%** |
| Best of all ten tested: Phi-4 14B | 100 | **56.0%** | **3%** |

The LLM experiment used the same balanced 100-item sample for every model; the rule-based baseline was evaluated on all 3,000 published training examples. Full model-by-model results are in [the LLM experiment report](experiments/README.md).

## Experiment 1: Rule-Based Baseline

A simple rule-based program achieved perfect scores on the published training set without using an LLM.

### How It Works

Attack statements follow a predictable pattern:

`Witness X says that Y is [attack expression]`

For example: `Witness Bob says that Alice is lying`.

The attack expression varies (`lying`, `not credible`, `dishonest`, etc.), but each puzzle gives us that expression in its opening rule.

The program simply:

1. Reads the attack expression from the rule.
2. Finds the **source (X)** and **target (Y)** of each attack using the statement pattern.
3. Builds the attack graph.
4. Applies the puzzle rule: a witness is believable if none of their attackers is believable.
5. Answers **Task 1** (is the witness making the factual claim believable?) and **Task 2** (which witnesses are believable?).

Example:

```text
Witness Alice says that the train is late
Witness Bob says that Alice is lying
Witness Charlie says that Bob is lying
```

Charlie is believable, so Bob is not. Since Bob is not believable, Alice is believable. The answers are **yes** for Task 1 and **Alice, Charlie** for Task 2.

**Finding:** The published training examples can be solved by matching text patterns and applying a graph algorithm. The program does not need to interpret the meaning of the statements.

## Experiment 2: Ten Small LLMs

We compared ten models using a **balanced sample of 100 training puzzles** (50 `yes` and 50 `no`). Each model received the original Task 1 and Task 2 prompts separately: **2,000 attempted calls** altogether. We used temperature `0`, no examples, and a maximum of 1,024 completion tokens.

The best model within the official 8B parameter cap was **Llama 3.1 8B Instruct** (50.8% Task 1 macro-F1, 1% Task 2 exact match). **Phi-4 14B** scored highest overall (56.0% and 3%), though it exceeds that cap.

See [Experiment 2: models, exact prompts, full scores, and instructions](experiments/README.md). Some models produced empty or invalid responses, which are reflected in the reported scores.

## How To Reproduce the Rule Baseline

Only Python 3 is needed; no packages or API keys.

**1. Run the unit tests:**

```bash
git clone https://github.com/Khalid-AlKhatib/PART.git
cd PART
python3 -m unittest discover -s tests -v
```

**2. Reproduce the training result:**

Download the [official training file](https://huggingface.co/datasets/michiel/argpart/tree/main/data), save it as `data/train.jsonl`, and run:

```bash
python3 argpart_rule_baseline.py data/train.jsonl
```

We obtained **3,000/3,000 correctly parsed**, **Task 1 macro-F1 = 1.0**, and **Task 2 exact-match accuracy = 100%**.

The rule baseline assumes the current statement templates and acyclic attack graphs. All results reported here are on the **published training set**, not the hidden test set.
