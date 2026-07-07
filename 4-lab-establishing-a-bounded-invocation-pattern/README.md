# Lab 4 - Establishing a Bounded Invocation Pattern

Supporting files for the "Establishing a Bounded Invocation Pattern" lab.

## Files

- `prompt.txt` contains the prompt submitted to the foundation model.
- `inference-profile.json` defines the bounded inference configuration.
- `invoke_bedrock.py` reads the prompt and inference profile, invokes Amazon Bedrock, and records invocation metrics.
- `results.csv` stores configuration values, token usage, and latency for each invocation.
