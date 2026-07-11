import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

MODEL_ID = "amazon.nova-pro-v1:0"

POLICY_FILE = Path("policy_config.json")
REQUESTS_FILE = Path("requests.csv")
RESULTS_FILE = Path("results.csv")

INFERENCE_CONFIG = {
    "maxTokens": 300,
    "temperature": 0.2,
    "topP": 0.9
}

SIMULATED_FAILURE_MARKER = "[SIMULATE_RETRY_FAILURE]"


def load_policy():
    with POLICY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_requests():
    with REQUESTS_FILE.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ensure_results_file():
    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0

    if not file_exists:
        with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "request_id",
                "decision",
                "attempts",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
                "response"
            ])


def evaluate_request(prompt, policy):
    """Apply the application-side policy layer before invoking Bedrock."""

    if len(prompt) > policy["maxPromptCharacters"]:
        return False, "limit_exceeded"

    normalized_prompt = prompt.lower()

    for blocked_term in policy["blockedTerms"]:
        if blocked_term.lower() in normalized_prompt:
            return False, "blocked"

    return True, "allowed"


def invoke_bedrock(client, prompt):
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


def invoke_with_retries(client, prompt, max_retries):
    attempts = 0
    total_start = time.perf_counter()

    while attempts <= max_retries:
        attempts += 1

        try:
            # This controlled marker guarantees that learners can observe
            # retry exhaustion without depending on an actual AWS outage.
            if SIMULATED_FAILURE_MARKER in prompt:
                raise RuntimeError("Simulated transient inference failure")

            response, _ = invoke_bedrock(client, prompt)

            total_latency_ms = round(
                (time.perf_counter() - total_start) * 1000
            )

            return response, attempts, total_latency_ms, None

        except (RuntimeError, BotoCoreError, ClientError) as error:
            print(f"  Attempt {attempts} failed: {error}")

            if attempts > max_retries:
                total_latency_ms = round(
                    (time.perf_counter() - total_start) * 1000
                )

                return None, attempts, total_latency_ms, error

            time.sleep(1)


def append_result(
    request_id,
    decision,
    attempts,
    response,
    latency_ms,
    response_text
):
    usage = response.get("usage", {}) if response else {}

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            request_id,
            decision,
            attempts,
            usage.get("inputTokens", 0),
            usage.get("outputTokens", 0),
            usage.get("totalTokens", 0),
            latency_ms,
            response_text
        ])


def main():
    policy = load_policy()
    requests = load_requests()
    ensure_results_file()

    client = boto3.client("bedrock-runtime")

    print("Running controlled invocation workflow...")
    print(f"Requests: {len(requests)}")
    print()

    for item in requests:
        request_id = item["request_id"]
        prompt = item["prompt"]

        print(f"Request {request_id}:")

        allowed, policy_decision = evaluate_request(prompt, policy)

        if not allowed:
            fallback_message = policy["fallbackMessages"][policy_decision]

            print(f"  Decision: {policy_decision}")
            print(f"  Response: {fallback_message}")

            append_result(
                request_id=request_id,
                decision=policy_decision,
                attempts=0,
                response=None,
                latency_ms=0,
                response_text=fallback_message
            )

            print()
            continue

        response, attempts, latency_ms, error = invoke_with_retries(
            client,
            prompt,
            policy["maxRetries"]
        )

        if response:
            response_text = (
                response["output"]["message"]["content"][0]["text"]
            )

            print("  Decision: allowed")
            print(f"  Attempts: {attempts}")
            print(f"  Latency: {latency_ms} ms")
            print("  Response returned by Amazon Bedrock")

            append_result(
                request_id=request_id,
                decision="allowed",
                attempts=attempts,
                response=response,
                latency_ms=latency_ms,
                response_text=response_text
            )

        else:
            fallback_message = (
                policy["fallbackMessages"]["inferenceFailed"]
            )

            print("  Decision: retry_exhausted")
            print(f"  Attempts: {attempts}")
            print(f"  Response: {fallback_message}")

            append_result(
                request_id=request_id,
                decision="retry_exhausted",
                attempts=attempts,
                response=None,
                latency_ms=latency_ms,
                response_text=fallback_message
            )

        print()

    print("Workflow complete. Review results.csv.")


if __name__ == "__main__":
    main()
