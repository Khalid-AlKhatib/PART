# PART: Small-Language-Model Reasoning Diagnostics

This repository contains **independent diagnostic experiments** for the [PART shared task](https://argpart.github.io/), not the official submission system. The [official website](https://argpart.github.io/#dataset) remains the authoritative source for submission rules.

PART uses **synthetic witness-testimony puzzles** generated from directed argumentation attack graphs. Task 1 asks whether a specified statement is believable (`yes`/`no`); Task 2 asks for the set of believable witnesses. These controlled puzzles can test how small language models interpret relations and apply a multi-step acceptance rule. They are **not** a realistic representation of human argumentation or evidence assessment.

In the published training data, the graphs are acyclic. Once the attacks are correctly extracted, computing acceptance is deterministic and relatively easy. A language model's end-to-end errors may therefore come from **relation extraction, multi-step reasoning, or answer formatting**; the final score alone does not identify which part failed.

## Finding 1: A Symbolic Baseline Solves the Training Data

We tested whether the [PART shared task](https://argpart.github.io/) can be solved without an LLM. A simple rule-based program achieved **100% on both tasks across all 3,000 published training examples**.

## How Does the Rule Baseline Work?

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

**Main finding:** The current training examples can be solved by matching text patterns and applying a graph algorithm. The program does not need to understand the meaning of the statements. This makes the released training set a weak discriminator of *unrestricted* computational methods, while leaving an interesting question about what a small language model can do internally.

## Finding 2: Ten Small-Model Prompting Experiments

We also completed a diagnostic comparison of ten language models on a **balanced sample of 100 published training puzzles**, using the original Task 1 and Task 2 prompts (2,000 attempted model calls in total). The best Task 1 macro-F1 among models at or below 8B was **50.8%**, and the highest Task 2 strict exact match in that group was **1%**. Several models produced errors or unusable outputs, so these figures are not definitive model capability rankings. See the exact prompt, per-model scores, and caveats in [`experiments/README.md`](experiments/README.md).

This comparison is **not an official PART submission**: three tested models exceed the official inference parameter cap, and official eligibility also depends on model licenses and API restrictions.

## Proposed LLM-Focused Methodology

Based on these findings, a precise research objective for PART would be: **Can small language models solve and generalize across structured argumentation puzzles directly from natural language, without outsourcing the decisions to a hand-coded parser or symbolic graph solver?**

We have documented a possible LLM-only inference policy in [`METHOD_ELIGIBILITY_PROPOSAL.md`](METHOD_ELIGIBILITY_PROPOSAL.md). It distinguishes **model-generated intermediate graphs and step-by-step reasoning** (allowed) from **external task-specific rules and executed graph solvers** (not allowed). It also proposes method disclosure and diagnostic evaluations by graph depth, language variation, and output compliance.

**Important: This is a discussion proposal, not an adopted rule.** The official PART website currently requires open-weight inference models totaling at most 8B parameters, restricts ensembles and mixture-of-experts by total parameters, and prohibits closed commercial APIs at every stage. It does **not** currently state an explicit ban on symbolic post-processing; the organizers would need to approve and publish any such change.

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

These results are for the **training set**, not the hidden test set. The program assumes the current statement templates and acyclic attack graphs.
