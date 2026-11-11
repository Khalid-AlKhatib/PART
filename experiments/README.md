# Experiment 2: Ten Language Models

Ten models were evaluated on 100 PART training examples on 11 November 2026. All received the same original task prompts.

- Models: [`models.json`](models.json)
- Code: [`run_small_llms.py`](run_small_llms.py)
- Workflow: [`small-llm-eval.yml`](../.github/workflows/small-llm-eval.yml)

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
- Zero-shot prompting, with no examples or graph-extraction hints. Gold labels were used for sampling and scoring, not included in the prompts.
- The code records raw answers, parsed predictions, errors, and token usage.

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

Empty or unparseable responses were counted as incorrect; a zero score with zero valid responses reflects failed requests, not a successful prediction. The three models over 8B were included for comparison.
