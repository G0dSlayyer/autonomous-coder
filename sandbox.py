"""All Docker execution lives here: one source of truth for image + limits.

  run_in_sandbox(script)        one script, one container     (retry loop, self-tests)
  run_job(job)                  N candidates x M inputs, ONE container   (consensus)
  run_hidden_batch(codes, prob) N candidates on hidden tests, ONE container
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = "python:3.11-slim"
MEMORY, CPUS, PIDS = "256m", "1", "128"


def _docker_cmd(d, tail, name=None, strict=True):
    """strict: read-only mount + read-only root fs (for batch jobs we fully control).
    not strict: exactly the flags your original run_in_sandbox used."""
    cmd = ["docker", "run", "--rm"]
    if name:
        cmd += ["--name", name]
    cmd += ["--network", "none", "--memory", MEMORY, "--cpus", CPUS, "--pids-limit", PIDS]
    if strict:
        cmd += ["--read-only", "--tmpfs", "/tmp:rw,size=64m", "-v", f"{d}:/work:ro", "-w", "/tmp"]
    else:
        cmd += ["-v", f"{d}:/work", "-w", "/work"]
    return cmd + [IMAGE] + tail


def hidden_script(code: str, problem: dict) -> str:
    return f"{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"


def run_in_sandbox(script: str) -> tuple[bool, str]:
    """Run a script in a locked-down, throwaway container. Returns (passed, output)."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as f:
            f.write(script)

        cmd = _docker_cmd(d, ["timeout", "10", "python", "run.py"], strict=False)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "Timed out."

        if r.returncode in (125, 126, 127):  # docker itself failed, not the code
            raise RuntimeError(f"Docker error:\n{r.stderr}")
        if r.returncode == 124:
            return False, "Timed out after 10s (possible infinite loop)."
        if r.returncode == 137:  # container killed (memory limit)
            return False, "Killed (out of memory?)."
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def _run_batch(files: dict, tail: list, timeout: int):
    """Write files into a temp dir, run one strict container, return last stdout line (parsed)."""
    name = f"sbx-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as d:
        for fname, content in files.items():
            with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
                f.write(content)
        try:
            r = subprocess.run(_docker_cmd(d, tail, name=name), capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return None
    if r.returncode != 0:
        raise RuntimeError(f"sandbox failed (code {r.returncode}): {r.stderr[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_job(job):
    """Consensus: job = {"entry_point", "candidates": {id: code}, "inputs": [[args], ...]}
    Returns {id: [[status, value?], ...]} (see harness.py)."""
    n_cand, n_in = len(job["candidates"]), len(job["inputs"])
    job.setdefault("per_candidate_timeout", 20)
    with open(os.path.join(HERE, "harness.py"), encoding="utf-8") as f:
        harness_src = f.read()
    res = _run_batch({"harness.py": harness_src, "job.json": json.dumps(job)},
                     ["python", "/work/harness.py", "/work/job.json"],
                     timeout=30 + job["per_candidate_timeout"] * n_cand)
    return res if res is not None else {c: [["TIMEOUT"]] * n_in for c in job["candidates"]}

_RUNNER = '''
import json, os, subprocess, sys
d, n = sys.argv[1], int(sys.argv[2])
out = []
for i in range(n):
    try:
        r = subprocess.run([sys.executable, os.path.join(d, f"c{i}.py")],
                           capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        out.append(["timeout", ""])
        continue
    if r.returncode == 0:
        out.append(["pass", ""])
    elif r.returncode < 0:
        out.append(["killed", ""])
    else:
        out.append(["fail", r.stderr.strip()[-500:]])
print(json.dumps(out))
'''


def classify(passed: bool, output: str) -> str:
    """Bucket one run: pass | wrong_answer | syntax_error | timeout | killed | runtime_error:<Name> | unknown"""
    if passed:
        return "pass"
    if output.startswith("Timed out"):
        return "timeout"
    if output.startswith("Killed"):
        return "killed"
    last = (output.strip().splitlines() or [""])[-1]
    name = last.split(":")[0].strip()
    if name == "AssertionError":
        return "wrong_answer"
    if name in ("SyntaxError", "IndentationError", "TabError"):
        return "syntax_error"
    if re.fullmatch(r"[A-Za-z_][\w.]*", name):
        return f"runtime_error:{name}"
    return "unknown"


def run_hidden_batch_kinds(codes, problem):
    """Score every candidate on HumanEval's hidden tests in ONE container.
    Returns one label per candidate: 'pass' or a failure kind (see classify)."""
    files = {f"c{i}.py": hidden_script(c, problem) for i, c in enumerate(codes)}
    files["runner.py"] = _RUNNER
    res = _run_batch(files, ["python", "/work/runner.py", "/work", str(len(codes))],
                     timeout=30 + 12 * len(codes))
    if res is None:
        return ["timeout"] * len(codes)
    return [status if status in ("pass", "timeout", "killed") else classify(False, text)
            for status, text in res]


def run_hidden_batch(codes, problem):
    """Same, as [bool, ...]."""
    return [k == "pass" for k in run_hidden_batch_kinds(codes, problem)]


if __name__ == "__main__":
    # Smoke test: 5 hand-written candidates, no LLM involved.
    job = {
        "entry_point": "add",
        "candidates": {
            "c0": "def add(a, b):\n    return a + b",
            "c1": "def add(a, b):\n    return b + a",
            "c2": "def add(a, b):\n    return a - b",
            "c3": "def add(a, b):\n    while True: pass",
            "c4": "def add(a, b):\n    raise ValueError('boom')",
        },
        "inputs": [[1, 2], [0, 0], [-1, 5]],
    }
    for cid, res in run_job(job).items():
        print(cid, res)
    # Expect: c0 == c1; c2 different values; c3 TIMEOUT; c4 ERR ValueError
    prob = {"test": "def check(f):\n    assert f(1, 2) == 3\n", "entry_point": "add"}
    print(run_hidden_batch(list(job["candidates"].values()), prob))  # [T, T, F, F, F]
