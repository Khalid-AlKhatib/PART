# Small LLM Experiment (Completed)

This experiment compared 10 models on PART's published training examples. It used the **original task prompts**, with no examples or graph-extraction hints. The manually triggered GitHub Actions run was completed on 11 November 2026.

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

## Results

Task 1 uses macro-F1; Task 2 uses strict exact match of the accepted-witness set. Scores are percentages on the 100 training examples. `Valid` means parsed answers (out of 100 per task); `Errors` counts failed requests (out of 200 per model).

| Model | Official 8B cap | Task 1 macro-F1 | Task 1 valid | Task 2 exact | Task 2 valid | Errors |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| LFM 3 1.7B | Within | 0.0% | 0 | 0% | 0 | 200 |
| Llama 3.2 3B Instruct | Within | 47.3% | 100 | 0% | 84 | 0 |
| Ministral 3B | Within | 35.0% | 100 | 0% | 85 | 0 |
| Gemma 3 4B | Within | 41.8% | 85 | 0% | 68 | 19 |
| Llama 3.1 8B Instruct | Within | **50.8%** | 71 | **1%** | 62 | 0 |
| Ministral 8B | Within | 40.5% | 100 | 0% | 90 | 0 |
| Granite 4.2 8B | Within | 0.0% | 0 | 0% | 0 | 200 |
| Qwen 3.5 9B | **Above** | 25.1% | 19 | 2% | 2 | 24 |
| Gemma 3 12B | **Above** | 45.6% | 83 | 0% | 92 | 0 |
| Phi-4 14B | **Above** | **56.0%** | 99 | **3%** | 67 | 0 |

Some models returned empty or unparseable answers, counted as incorrect. The 1,024-token limit may have affected reasoning models, particularly those with zero valid responses. The three models above 8B were included for comparison. All results are from the published training set, using one zero-shot prompt configuration.

## How To Re-run

1. Add your OpenRouter key as a GitHub Actions secret named `OPENROUTER_API_KEY` in [PART Settings](https://github.com/Khalid-AlKhatib/PART/settings/secrets/actions). Do not commit the key or paste it in an issue.
2. Open [GitHub Actions](https://github.com/Khalid-AlKhatib/PART/actions) and select **PART Small LLM Evaluation**.
3. Choose **Run workflow**, set `confirm` to `RUN`, and optionally choose 10 (pilot), 50, or 100 examples.
4. Download the results from the workflow's **Artifacts** section. New runs incur OpenRouter API costs.

The runner downloads the public training file from [Hugging Face](https://huggingface.co/datasets/michiel/argpart/tree/main/data). Results are evaluations on the *published training set*, not the hidden test set. Model availability and pricing can change; the runner checks availability before paid requests.
