import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

MODEL_ID = "amazon.nova-pro-v1:0"
RESULTS_FILE = Path("results.csv")

PROMPT = "Describe the benefits of cloud computing for a growing retail business."

PROFILE_FILES = [
    Path("conservative.json"),
    Path("balanced.json"),
    Path("expressive.json")
]


def read_profile(profile_file):
    with profile_file.open("r") as file:
        return json.load(file)


def ensure_results_file():
    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0

    if not file_exists:
        with RESULTS_FILE.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "profile_name",
                "model_id",
                "max_tokens",
                "temperature",
                "top_p",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms"
            ])


def invoke_model(client, inference_config):
    start_time = time.perf_counter()

    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": PROMPT
                    }
                ]
            }
        ],
        inferenceConfig=inference_config
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000)
    return response, latency_ms


def append_result(profile, response, latency_ms):
    usage = response.get("usage", {})
    inference_config = profile["inferenceConfig"]

    with RESULTS_FILE.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            profile["profileName"],
            MODEL_ID,
            inference_config["maxTokens"],
            inference_config["temperature"],
            inference_config["topP"],
            usage.get("inputTokens"),
            usage.get("outputTokens"),
            usage.get("totalTokens"),
            latency_ms
        ])


def main():
    client = boto3.client("bedrock-runtime")
    ensure_results_file()

    print("Comparing predefined inference configurations...")
    print(f"Prompt: {PROMPT}")
    print()

    for profile_file in PROFILE_FILES:
        profile = read_profile(profile_file)
        inference_config = profile["inferenceConfig"]

        print(f"Running profile: {profile['profileName']}")
        print(f"  Description: {profile['description']}")
        print(f"  Config: {json.dumps(inference_config)}")

        response, latency_ms = invoke_model(client, inference_config)
        append_result(profile, response, latency_ms)

        usage = response.get("usage", {})
        output_text = response["output"]["message"]["content"][0]["text"]

        print(f"  Input tokens:  {usage.get('inputTokens')}")
        print(f"  Output tokens: {usage.get('outputTokens')}")
        print(f"  Total tokens:  {usage.get('totalTokens')}")
        print(f"  Latency:       {latency_ms} ms")
        print()
        print("  Response preview:")
        print("  " + output_text[:250].replace("\n", "\n  "))
        print()

    print("Comparison complete. Review results.csv to compare profile behavior.")


if __name__ == "__main__":
    main()
