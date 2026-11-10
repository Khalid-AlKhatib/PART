#!/usr/bin/env python3
"""Zero-shot PART evaluation through OpenRouter; standard library only.

No request is sent without OPENROUTER_API_KEY. The GitHub Actions workflow
is manually triggered and requires a separate confirmation input.
"""

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
SYSTEM_PROMPT = (
    "Solve the reasoning puzzle. Follow its rules carefully.\n"
    "Return only the final answer in the format requested\n"
    "at the end of the puzzle. Do not include an explanation."
)
TASK_FIELDS = {"binary": "binary_prompt", "multi_label": "multi_label_prompt"}


def select_examples(records, sample_size, seed):
    """Choose an equal number of positive and negative examples, without replacement."""
    if sample_size < 2 or sample_size % 2:
        raise ValueError("sample size must be an even number of at least 2")
    rng = random.Random(seed)
    by_label = {"yes": [], "no": []}
    for record in records:
        if record.get("binary") in by_label and "multi_label" in record:
            by_label[record["binary"]].append(record)
    half = sample_size // 2
    if any(len(by_label[label]) < half for label in by_label):
        raise ValueError("not enough labeled examples for balanced sampling")
    sample = rng.sample(by_label["yes"], half) + rng.sample(by_label["no"], half)
    rng.shuffle(sample)
    return sample


def get_speaker_names(prompt):
    return re.findall(r"^Witness (.+?) says that ", prompt, flags=re.MULTILINE)


def parse_binary_answer(raw):
    # A copied instruction such as "Answer: yes or no" is not a prediction.
    matches = re.findall(
        r"^\s*Answer\s*:\s*(yes|no)\s*[.!]?\s*$",
        raw, flags=re.IGNORECASE | re.MULTILINE,
    )
    if matches:
        return matches[-1].lower()
    short = raw.strip().lower().strip(". ")
    return short if short in {"yes", "no"} else None


def parse_list_answer(raw, speaker_names):
    # The official prompt requests "Answer: ['Alice', 'Bob']". Accept both
    # single- and double-quoted lists; retain invalid answers as failures.
    text = raw.rsplit("Answer:", 1)[-1] if "Answer:" in raw else raw
    matches = re.findall(r"\[[^\]]*\]", text, flags=re.DOTALL)
    if not matches:
        return None
    try:
        names = ast.literal_eval(matches[-1])
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return None
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        return None
    canonical = {name.casefold(): name for name in speaker_names}
    normalized = []
    for name in names:
        matched = canonical.get(name.strip().casefold())
        if matched is None:
            return None
        normalized.append(matched)
    return sorted(set(normalized))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def http_json(url, *, body=None, key=None, timeout=90):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    if body:
        headers["HTTP-Referer"] = "https://github.com/Khalid-AlKhatib/PART"
        headers["X-Title"] = "PART small LLM evaluation"
    request = Request(url, data=data, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_models_available(models):
    catalog = http_json(MODELS_URL)
    available = {entry.get("id") for entry in catalog.get("data", [])}
    missing = [model["id"] for model in models if model["id"] not in available]
    if missing:
        raise RuntimeError("Models missing from current OpenRouter catalog: " + ", ".join(missing))


def ask_openrouter(model_id, prompt, api_key, max_tokens):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        try:
            return http_json(API_URL, body=payload, key=api_key)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            if attempt == 2:
                raise RuntimeError("OpenRouter connection or timeout error") from exc
        time.sleep(2 ** attempt)
    raise RuntimeError("Request retries exhausted")


def evaluate_one(model, record, task, api_key, max_tokens):
    start = time.monotonic()
    raw = None
    usage = {}
    error = None
    try:
        response = ask_openrouter(model["id"], record[TASK_FIELDS[task]], api_key, max_tokens)
        message = response["choices"][0]["message"]
        content = message.get("content")
        raw = content if isinstance(content, str) else ""
        usage = response.get("usage") or {}
        if not raw:
            error = "Empty response"
    except (RuntimeError, KeyError, IndexError, TypeError, ValueError) as exc:
        error = str(exc)
    if raw is None:
        prediction = None
    elif task == "binary":
        prediction = parse_binary_answer(raw)
    else:
        prediction = parse_list_answer(raw, get_speaker_names(record["multi_label_prompt"]))
    return {
        "model": model["id"], "task": task, "id": record["id"],
        "gold": record[task], "prediction": prediction, "raw_response": raw,
        "valid_answer": prediction is not None,
        "error": error, "usage": usage,
        "seconds": round(time.monotonic() - start, 3),
    }


def class_f1(rows, label):
    tp = sum(row["gold"] == label and row["prediction"] == label for row in rows)
    fp = sum(row["gold"] != label and row["prediction"] == label for row in rows)
    fn = sum(row["gold"] == label and row["prediction"] != label for row in rows)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def summarize(rows):
    binary = [row for row in rows if row["task"] == "binary"]
    multi = [row for row in rows if row["task"] == "multi_label"]
    scores = {"total_requests": len(rows),
              "request_errors": sum(bool(row["error"]) for row in rows)}
    if binary:
        scores.update({
            "task1_n": len(binary),
            "task1_valid_answers": sum(row["valid_answer"] for row in binary),
            "task1_accuracy": sum(row["prediction"] == row["gold"] for row in binary) / len(binary),
            "task1_macro_f1": (class_f1(binary, "yes") + class_f1(binary, "no")) / 2,
        })
    if multi:
        tp = fp = fn = exact = 0
        instance_f1 = []
        for row in multi:
            gold = set(row["gold"])
            predicted = set(row["prediction"] or [])
            tp += len(gold & predicted)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            exact += row["valid_answer"] and predicted == gold
            denom = len(gold) + len(predicted)
            instance_f1.append((2 * len(gold & predicted) / denom if denom else 1.0)
                               if row["valid_answer"] else 0.0)
        scores.update({
            "task2_n": len(multi),
            "task2_valid_answers": sum(row["valid_answer"] for row in multi),
            "task2_exact_match": exact / len(multi),
            "task2_micro_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
            "task2_mean_instance_f1": sum(instance_f1) / len(multi),
        })
    prices = [row["usage"].get("cost") for row in rows if isinstance(row.get("usage"), dict)]
    observed_prices = [float(cost) for cost in prices if isinstance(cost, (int, float))]
    if observed_prices:
        scores["reported_cost_usd"] = round(sum(observed_prices), 5)
        scores["requests_with_reported_cost"] = len(observed_prices)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("models.json"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("results/small_llms"))
    parser.add_argument("--dry-run", action="store_true", help="Prepare sample without any API calls")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")

    models = json.loads(args.models.read_text(encoding="utf-8"))
    if len(models) != 10 or len({model["id"] for model in models}) != 10:
        parser.error("model configuration must contain 10 distinct models")
    sample = select_examples(read_jsonl(args.data), args.sample_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "models": models, "sample_ids": [row["id"] for row in sample],
        "sample_size": len(sample), "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_fields": TASK_FIELDS,
        "temperature": 0, "max_tokens": args.max_tokens,
        "dry_run": args.dry_run,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Dry run: {len(models)} models x {len(sample)} examples x 2 tasks; no API calls.")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        parser.error("OPENROUTER_API_KEY is missing (set it as a GitHub Actions secret)")
    check_models_available(models)  # Abort before paid calls if a model has disappeared.

    pending = [(model, record, task) for model in models for record in sample for task in TASK_FIELDS]
    collected = []
    by_model = Counter()
    with (args.output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_one, model, record, task, api_key, args.max_tokens)
                       for model, record, task in pending]
            for future in as_completed(futures):
                row = future.result()
                collected.append(row)
                by_model[row["model"]] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if len(collected) % 100 == 0:
                    print(f"Completed {len(collected)}/{len(pending)} calls", flush=True)

    by_id = {}
    for model in models:
        rows = [row for row in collected if row["model"] == model["id"]]
        by_id[model["id"]] = summarize(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(by_id, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(by_id, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
