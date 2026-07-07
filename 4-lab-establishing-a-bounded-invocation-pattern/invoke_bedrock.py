import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

MODEL_ID = "amazon.nova-pro-v1:0"
PROMPT_FILE = Path("prompt.txt")
PROFILE_FILE = Path("inference-profile.json")
RESULTS_FILE = Path("results.csv")


def read_prompt():
    return PROMPT_FILE.read_text().strip()


def read_inference_profile():
    with PROFILE_FILE.open("r") as file:
        return json.load(file)


def write_result(response, latency_ms, inference_config):
    usage = response.get("usage", {})
    output_text = response["output"]["message"]["content"][0]["text"]

    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0

    with RESULTS_FILE.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "model_id",
                "max_tokens",
                "temperature",
                "top_p",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            MODEL_ID,
            inference_config["maxTokens"],
            inference_config["temperature"],
            inference_config["topP"],
            usage.get("inputTokens"),
            usage.get("outputTokens"),
            usage.get("totalTokens"),
            latency_ms
        ])

    return output_text


def main():
    prompt = read_prompt()
    inference_config = read_inference_profile()

    client = boto3.client("bedrock-runtime")

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
        inferenceConfig=inference_config
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    output_text = write_result(response, latency_ms, inference_config)

    print("Model response:")
    print(output_text)
    print()
    print("Invocation recorded in results.csv")


if __name__ == "__main__":
    main()
