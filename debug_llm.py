from backend.main import app
from backend.llm.client import create_client_from_mapping, LLMRequest

client = create_client_from_mapping(app.state.config)

print("=== CONFIG ===")
print("provider =", client.config.provider)
print("model =", client.config.model)
print("structured =", client.config.enable_structured_output)
print("ollama_mode =", repr(client.config.ollama_structured_output_mode))
print("base_url =", client.config.base_url)

request = LLMRequest(
    task="qa",
    messages=[
        {
            "role": "user",
            "content": 'Return ONLY valid JSON. No markdown. Return exactly {"answer":"test"}',
        }
    ],
    response_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
)

print("\n=== CALLING LLM ===")

try:
    response = client.generate(request)

    print("SUCCESS")
    print("type =", type(response).__name__)
    print("text =", repr(response.text))
    print("model =", response.model)
    print("finish_reason =", response.finish_reason)
    print("metadata =", response.metadata)

except Exception as exc:
    print("FAILED")
    print("exception_type =", type(exc).__name__)
    print("exception =", repr(exc))
    raise
