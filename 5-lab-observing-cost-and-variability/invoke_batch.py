import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

MODEL_ID = "amazon.nova-pro-v1:0"
PROMPT_DATASET_FILE = Path("prompt_dataset.csv")
RESULTS_FILE = Path("results.csv")

INFERENCE_CONFIG = {
    "maxTokens": 300,
    "temperature": 0.2,
    "topP": 0.9
}


def read_prompts():
    with PROMPT_DATASET_FILE.open("r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def ensure_results_file():
    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0

    if not file_exists:
        with RESULTS_FILE.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "prompt_id",
                "model_id",
                "max_tokens",
                "temperature",
                "top_p",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms"
            ])


def invoke_model(client, prompt):
    start_time = time.perf_counter()

    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        inferenceConfig=INFERENCE_CONFIG
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    return response, latency_ms


def append_result(prompt_id, response, latency_ms):
    usage = response.get("usage", {})

    with RESULTS_FILE.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            prompt_id,
            MODEL_ID,
            INFERENCE_CONFIG["maxTokens"],
            INFERENCE_CONFIG["temperature"],
            INFERENCE_CONFIG["topP"],
            usage.get("inputTokens"),
            usage.get("outputTokens"),
            usage.get("totalTokens"),
            latency_ms
        ])


def main():
    prompts = read_prompts()
    ensure_results_file()

    client = boto3.client("bedrock-runtime")

    print("Running controlled prompt dataset...")
    print(f"Prompts: {len(prompts)}")
    print()

    for item in prompts:
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]

        print(f"Running prompt {prompt_id}...")

        response, latency_ms = invoke_model(client, prompt)
        append_result(prompt_id, response, latency_ms)

        usage = response.get("usage", {})
        print(f"  Input tokens:  {usage.get('inputTokens')}")
        print(f"  Output tokens: {usage.get('outputTokens')}")
        print(f"  Total tokens:  {usage.get('totalTokens')}")
        print(f"  Latency:       {latency_ms} ms")
        print()

    print("Batch complete. Review results.csv to compare token usage and latency.")


if __name__ == "__main__":
    main()
