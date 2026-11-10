# PART: Rule-Based Baseline

We tested whether the [PART shared task](https://argpart.github.io/) can be solved without an LLM. A simple rule-based program achieved **100% on both tasks across all 3,000 published training examples**.

## How Does It Work?

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

**Main finding:** The current training examples can be solved by matching text patterns and applying a graph algorithm. The program does not need to understand the meaning of the statements.

## How To Test

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

These results are for the **training set**, not the hidden test set. The program assumes the current statement templates and acyclic attack graphs.
