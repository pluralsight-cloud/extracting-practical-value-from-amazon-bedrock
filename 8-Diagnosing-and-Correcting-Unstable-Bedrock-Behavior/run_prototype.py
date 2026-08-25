import csv
import json
import time
from pathlib import Path

import boto3

MODEL_ID = "amazon.nova-pro-v1:0"

WORKLOAD_FILE = Path("workload.csv")
PROFILES_FILE = Path("profiles.json")
CONFIG_FILE = Path("prototype_config.json")
VALIDATION_FILE = Path("validation.json")

TOKEN_THRESHOLD = 1200


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_workload():
    with WORKLOAD_FILE.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def invoke_model(client, prompt, inference_config):
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

    return response, latency_ms


def main():
    profiles = load_json(PROFILES_FILE)
    config = load_json(CONFIG_FILE)
    workload = load_workload()

    active_profile = config["active_profile"]
    inference_config = profiles[active_profile]

    client = boto3.client("bedrock-runtime")

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    successful_invocations = 0

    print("Evaluating prototype behavior...")
    print(f"Active profile: {active_profile}")
    print(f"Inference configuration: {inference_config}")
    print()

    for item in workload:
        request_id = item["request_id"]
        prompt = item["prompt"]

        print(f"Running request {request_id}...")

        response, latency_ms = invoke_model(
            client,
            prompt,
            inference_config
        )

        usage = response.get("usage", {})

        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        request_total_tokens = usage.get("totalTokens", 0)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += request_total_tokens
        successful_invocations += 1

        print(f"  Input tokens:  {input_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  Total tokens:  {request_total_tokens}")
        print(f"  Latency:       {latency_ms} ms")
        print()

    token_usage_status = (
        "PASS"
        if total_output_tokens <= TOKEN_THRESHOLD
        else "FAIL"
    )

    overall_status = (
        "PASS"
        if token_usage_status == "PASS"
        else "NEEDS_ATTENTION"
    )

    validation = {
        "status": overall_status,
        "active_profile": active_profile,
        "invocation_count": successful_invocations,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "token_threshold": TOKEN_THRESHOLD,
        "token_usage_status": token_usage_status
    }

    with VALIDATION_FILE.open("w", encoding="utf-8") as file:
        json.dump(validation, file, indent=2)

    print("Prototype evaluation complete.")
    print()
    print(f"Status: {overall_status}")
    print(f"Active profile: {active_profile}")
    print(f"Invocation count: {successful_invocations}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Token threshold: {TOKEN_THRESHOLD}")
    print(f"Token usage status: {token_usage_status}")
    print()
    print("Validation results written to validation.json")


if __name__ == "__main__":
    main()
