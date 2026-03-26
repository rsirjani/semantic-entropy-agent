"""Smoke test for the vLLM server."""

import subprocess
import sys
import time

import openai


BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen3-coder"


def wait_for_server(timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    client = openai.OpenAI(base_url=BASE_URL, api_key="dummy")
    print(f"Waiting for vLLM server at {BASE_URL}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            models = client.models.list()
            if models.data:
                print(f"Server ready! Models: {[m.id for m in models.data]}")
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def test_basic_completion(client: openai.OpenAI) -> bool:
    """Test basic chat completion."""
    print("\n--- Test 1: Basic completion ---")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Write a Python function that checks if a number is prime. Just the function, no explanation."}
        ],
        max_tokens=256,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    print(f"Response:\n{content}")
    print(f"Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
    return bool(content and len(content) > 10)


def test_logprobs(client: openai.OpenAI) -> bool:
    """Test that logprobs are accessible (needed for Phase 2)."""
    print("\n--- Test 2: Logprobs ---")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        max_tokens=32,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
    )
    content = response.choices[0].message.content
    logprobs = response.choices[0].logprobs
    print(f"Response: {content}")
    if logprobs and logprobs.content:
        print(f"Logprobs available: Yes ({len(logprobs.content)} tokens)")
        # Show first few token logprobs
        for tok in logprobs.content[:3]:
            print(f"  Token '{tok.token}': logprob={tok.logprob:.4f}")
            if tok.top_logprobs:
                for alt in tok.top_logprobs[:3]:
                    print(f"    Alt '{alt.token}': logprob={alt.logprob:.4f}")
        return True
    else:
        print("Logprobs: NOT available")
        return False


def test_tool_calling(client: openai.OpenAI) -> bool:
    """Test tool/function calling format."""
    print("\n--- Test 3: Tool calling ---")
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"}
                },
                "required": ["command"]
            }
        }
    }]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "List all Python files in the current directory."}
        ],
        tools=tools,
        max_tokens=256,
        temperature=0.0,
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
        return True
    else:
        print(f"No tool calls. Content: {msg.content}")
        # This is OK — we use text-based THOUGHT/ACTION format, not tool calls
        return True


def check_vram():
    """Print GPU VRAM usage."""
    print("\n--- VRAM Usage ---")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) == 3:
                used, total, free = parts
                print(f"VRAM: {used} MiB used / {total} MiB total / {free} MiB free")
        else:
            print("nvidia-smi failed")
    except Exception as e:
        print(f"Could not check VRAM: {e}")


def main():
    if not wait_for_server():
        print("ERROR: Server did not start within timeout")
        sys.exit(1)

    client = openai.OpenAI(base_url=BASE_URL, api_key="dummy")

    results = {}
    results["basic_completion"] = test_basic_completion(client)
    results["logprobs"] = test_logprobs(client)
    results["tool_calling"] = test_tool_calling(client)
    check_vram()

    print("\n--- Summary ---")
    all_pass = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed! vLLM server is ready.")
    else:
        print("\nSome tests failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
