# Small LLM Experiment (Not Run Yet)

This experiment compares 10 small models on PART's published training examples. It uses the **original task prompts**, with no examples or graph-extraction hints.

- Models: [`models.json`](models.json)
- Code: [`run_small_llms.py`](run_small_llms.py)
- GitHub Actions: [manual workflow](../.github/workflows/small-llm-eval.yml)

**System prompt for both tasks:**

```text
Solve the reasoning puzzle. Follow its rules carefully.
Return only the final answer in the format requested
at the end of the puzzle. Do not include an explanation.
```

**User prompt:** The unmodified `binary_prompt` (Task 1) or `multi_label_prompt` (Task 2) field from the dataset.

## Design

- Same 100 examples for all models (50 `yes`, 50 `no`; random seed 42).
- Each task is sent separately: **2,000 API calls** for 10 models × 100 examples × 2 tasks.
- Temperature 0 and maximum 1,024 completion tokens per call.
- Outputs: raw responses, parsed answers, errors, token usage, and a summary with Task 1 accuracy/macro-F1, Task 2 exact match/F1, and parse success rates.
- No model receives gold answers. Labels are used only to select a balanced sample and evaluate predictions.
- The workflow runs **only when manually triggered**. Committing it does not make paid calls.

## Run After Approval

1. Add your OpenRouter key as a GitHub Actions secret named `OPENROUTER_API_KEY` in [PART Settings](https://github.com/Khalid-AlKhatib/PART/settings/secrets/actions). Do not commit the key or paste it in an issue.
2. Open [GitHub Actions](https://github.com/Khalid-AlKhatib/PART/actions) and select **PART Small LLM Evaluation**.
3. Choose **Run workflow**, set `confirm` to `RUN`, and optionally choose 10 (pilot), 50, or 100 examples.
4. Download the results from the workflow's **Artifacts** section.

The runner downloads the public training file from [Hugging Face](https://huggingface.co/datasets/michiel/argpart/tree/main/data). Results are evaluations on the *published training set*, not the hidden test set. Model availability and pricing can change; the runner checks availability before paid requests.
