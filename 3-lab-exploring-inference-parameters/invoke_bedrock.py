import boto3

client = boto3.client("bedrock-runtime")

response = client.converse(
    modelId="amazon.nova-pro-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "Explain cloud computing in three bullet points."
                }
            ]
        }
    ],
    inferenceConfig={
        "maxTokens": 300,
        "temperature": 0.2,
        "topP": 0.9
    }
)

print(response["output"]["message"]["content"][0]["text"])