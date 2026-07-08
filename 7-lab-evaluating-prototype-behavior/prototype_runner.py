import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

MODEL_ID = "amazon.nova-pro-v1:0"

WORKLOAD_FILE = Path("workload.csv")
METRICS_FILE = Path("metrics.csv")

INFERENCE_CONFIG = {
    "maxTokens": 400,
    "temperature": 0.5,
    "topP": 0.9
}

OUTPUT_LENGTH_THRESHOLD = 300


def read_workload():
    with WORKLOAD_FILE.open("r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def ensure_metrics_file():
    file_exists = (
        METRICS_FILE.exists()
        and METRICS_FILE.stat().st_size > 0
    )

    if not file_exists:
        with METRICS_FILE.open("w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "request_id",
                "request_type",
                "model_id",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
                "output_length_flag",
                "status"
            ])


def invoke_model(client, prompt):
    start_time = time.perf_counter()

    try:
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

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000
        )

        return response, latency_ms, "success"

    except Exception as error:
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000
        )

        print(f"  Invocation failed: {error}")

        return None, latency_ms, "failed"


def append_metric(item, response, latency_ms, status):
    if response:
        usage = response.get("usage", {})

        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        total_tokens = usage.get("totalTokens", 0)

        output_length_flag = (
            "high"
            if output_tokens >= OUTPUT_LENGTH_THRESHOLD
            else "normal"
        )

    else:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        output_length_flag = "not_available"

    with METRICS_FILE.open("a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            item["request_id"],
            item["request_type"],
            MODEL_ID,
            input_tokens,
            output_tokens,
            total_tokens,
            latency_ms,
            output_length_flag,
            status
        ])


def main():
    workload = read_workload()
    ensure_metrics_file()

    client = boto3.client("bedrock-runtime")

    print("Running prototype workload...")
    print(f"Requests: {len(workload)}")
    print()

    for item in workload:
        print(
            f"Request {item['request_id']} "
            f"({item['request_type']})..."
        )

        response, latency_ms, status = invoke_model(
            client,
            item["prompt"]
        )

        append_metric(
            item,
            response,
            latency_ms,
            status
        )

        if response:
            usage = response.get("usage", {})

            output_tokens = usage.get(
                "outputTokens",
                0
            )

            print(
                f"  Output tokens: {output_tokens}"
            )

            print(
                f"  Latency: {latency_ms} ms"
            )

            if output_tokens >= OUTPUT_LENGTH_THRESHOLD:
                print(
                    "  Flag: output length is above "
                    "the expected threshold"
                )

        print(f"  Status: {status}")
        print()

    print("Prototype workload complete.")
    print("Review metrics.csv for behavior patterns.")


if __name__ == "__main__":
    main()
