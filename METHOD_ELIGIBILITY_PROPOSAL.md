# Proposed LLM-Only Method Eligibility for PART

> **Discussion draft — 11 November 2026.** This is **not an adopted submission rule**. Only the [official PART website](https://argpart.github.io/#dataset) defines current eligibility. The organizers would need to agree on and publish any new restriction before enforcing it.

## Scientific Objective

PART's synthetic witness-testimony puzzles provide a controlled test of whether **small language models can extract attack relations from text and apply the witness-acceptance rule across multiple steps**. The intention of a possible LLM-only restriction would be to evaluate those capabilities *inside the model*, not the accuracy of an arbitrary pipeline that includes hand-coded graph algorithms.

This restriction would narrow the task's scientific claim: it would evaluate structured reasoning from natural language by small LLMs, **not** general methods for computational argumentation or realistic evidence assessment. The current training set can already be solved perfectly by a template parser followed by a simple graph traversal; see the [reproducible baseline](README.md).

## Proposed Boundary at Inference Time

**Principle:** An eligible language model must perform the substantive, task-specific interpretation and reasoning that determines each prediction. Generic input/output handling is fine; a separate task-specific parser or symbolic solver is not.

| Method or component | Proposed status | Explanation |
| --- | --- | --- |
| Zero-/few-shot prompting, reasoning prompts, self-reflection | **Allowed** | The model computes the answer. |
| Fine-tuning, parameter-efficient tuning, training on generated examples, distillation | **Allowed** subject to official model/API restrictions | The model learns to produce the answer. |
| Multiple inference passes, self-consistency, model ensembles | **Allowed** subject to the official total-parameter cap | Substantive predictions remain model-generated. |
| An LLM writes an intermediate attack list or graph **and reasons about it itself** | **Allowed** | A graph written as model output is not an external graph solver. |
| Generic tokenization, whitespace cleanup, extracting the model's final answer into the required JSONL fields | **Allowed** | These do not change which witnesses are accepted. |
| Hand-written regexes or templates that extract attacks or witnesses from puzzle text | **Not allowed** | They replace model-based relation extraction. |
| Executed attack-graph traversal, grounded-semantics algorithm, or other external graph solver | **Not allowed** | It performs the decisive inference outside the model. |
| External SAT/ASP/Prolog/rule engines or executed model-generated code that computes the answer | **Not allowed** | A tool, rather than the model, solves the puzzle. |
| Hard-coded task-specific label heuristics, answer lookup tables | **Not allowed** | They bypass the model's reasoning. |

**Important distinction:** A prompt may explain the acceptance rule and request step-by-step reasoning; that is still model-based inference. A prompt may also ask a model to output a graph as an intermediate text representation. What would be excluded is **feeding that graph to separate code that calculates the accepted witnesses**.

Example:

- **Allowed:** An eligible LLM outputs `Bob attacks Alice`, then itself concludes that Alice is rejected and returns the final answer.
- **Not allowed:** An LLM outputs `Bob attacks Alice`, and a Python function parses the edge and runs the acceptance algorithm to produce the final answer.

For ambiguous cases (e.g., specialized decoding procedures or intermediate representations), organizers should publish examples and a consistent interpretation before submissions begin.

## Preserve Existing Official Restrictions

The [published rules](https://argpart.github.io/#dataset) currently state:

- Inference models must be **open-weight and at most 8B parameters in total**.
- Mixture-of-experts count by **total** parameters; ensembles count by the **sum** of model weights.
- Larger **open-weight** models may be used at intermediate stages (e.g., distillation).
- The model weights must be releasable under an appropriate license allowing unrestricted non-commercial use (see official wording).
- **Closed commercial APIs are prohibited at every stage**, including data generation and distillation.
- At most **three runs per subtask** may be submitted.

This proposal changes **none** of those limits. In particular, a model trained using a prohibited closed commercial API would not become eligible simply because its final inference model is small.

## Proposed Transparency Requirements

If this methodological rule is adopted, team reports should identify:

1. Exact inference models, parameter counts, licenses, and any fine-tuning or distillation.
2. Training data and any synthetic-data generators, including models used during intermediate stages.
3. Prompts, inference passes, self-consistency or ensemble procedures, tools, and post-processing.
4. Any model-generated graph or structured representation, and **which component computes the final prediction**.
5. Reproducibility materials (code, prompts, and weights where possible), with a clear process for organizer verification.

**Enforcement limitation:** The prediction JSONL files alone cannot prove that a system used only LLM inference. Eligibility would require system disclosure and, where needed, code inspection or reproduction. This limitation should be acknowledged publicly.

## Recommended Diagnostic Evaluation

The current end-to-end score combines several abilities. To avoid claiming that every error is a *reasoning* error, analyze them separately:

- **Relation extraction:** Can the LLM identify who attacks whom?
- **Reasoning with known relations:** If the correct attack list is provided, can it infer the accepted witnesses?
- **Output compliance:** Can it produce valid yes/no and witness-set answers?

Also report performance by attack-chain depth, graph size and shape, testimony order, distractors, and unseen wording. A split by graph structure and reasoning depth would better test systematic generalization than a single aggregate score. The official website already warns that test conditions may differ from training; this proposal recommends **explicitly measuring** which differences matter.

**Open organizer decision:** Restrict the main competition to LLM-based inference, or offer a separate unrestricted track. Given the perfect symbolic baseline on the current training set, an unrestricted track might offer little differentiation unless the task changes substantially.
