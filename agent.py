import os
import subprocess

from providers import get_provider, extract_code

# Pick the backend without editing code:  PROVIDER=ollama:qwen2.5-coder:7b python agent.py
PROVIDER = os.getenv("PROVIDER", "groq:openai/gpt-oss-20b")
provider = get_provider(PROVIDER)
MAX_ATTEMPTS = 5

TASK_DIR = "task"
SOLUTION_PATH = os.path.join(TASK_DIR, "solution.py")
TEST_PATH = os.path.join(TASK_DIR, "test_palindrome.py")

SYSTEM_PROMPT = """You are an autonomous coding agent. You will be given a task and, \
possibly, the output of a failed test run. Your job is to write a single Python file \
that makes the tests pass.

Rules:
- Respond with ONLY a Python code block containing the full contents of solution.py.
- Do not include explanations, markdown headers, or anything outside the code block.
- The file must define exactly what the tests import and use.
"""


def read_test_file():
    with open(TEST_PATH, "r") as f:
        return f.read()


def write_solution(code: str):
    with open(SOLUTION_PATH, "w") as f:
        f.write(code)


def run_tests() -> tuple[bool, str]:
    """Runs pytest inside a throwaway Docker container, returns (passed, output)."""
    task_abs_path = os.path.abspath(TASK_DIR)

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{task_abs_path}:/task",
            "-w", "/task",
            "python:3.11-slim",
            "sh", "-c", "pip install pytest -q && pytest . -v",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


def ask_llm(messages: list[dict]) -> str:
    return provider.chat(messages, temperature=0.2).text


def main():
    test_code = read_test_file()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here are the tests:\n\n```python\n{test_code}\n```\n\nWrite solution.py."},
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'='*50}")
        print(f"Attempt {attempt}/{MAX_ATTEMPTS}  ({provider.name})")
        print(f"{'='*50}")

        response_text = ask_llm(messages)
        code = extract_code(response_text)

        print("\n--- Generated code ---")
        print(code)

        write_solution(code)

        passed, test_output = run_tests()

        print("\n--- Test output ---")
        print(test_output)

        if passed:
            print(f"\n✓ SUCCESS on attempt {attempt}")
            return

        # Feed the failure back to the model for the next attempt
        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": f"The tests failed. Here is the output:\n\n{test_output}\n\nFix solution.py.",
        })

    print(f"\n✗ FAILED after {MAX_ATTEMPTS} attempts")


if __name__ == "__main__":
    main()
