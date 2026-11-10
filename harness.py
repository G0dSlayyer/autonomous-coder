"""Runs INSIDE the sandbox container (and is imported on the host for `norm`).

Usage in container:  python harness.py /work/job.json
job.json = {"entry_point": str, "candidates": {id: code}, "inputs": [[args...], ...]}
Prints one JSON line: {id: [[status, value?], ...one per input]}
Statuses: OK (value = repr of normalized output), ERR (value = exception name),
          TIMEOUT, FLAKY (candidate disagreed with itself across two runs).
"""
import copy
import json
import os
import signal
import subprocess
import sys


def norm(x):
    """Make outputs comparable: round floats, order sets."""
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, tuple):
        return tuple(norm(i) for i in x)
    if isinstance(x, list):
        return [norm(i) for i in x]
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, (set, frozenset)):
        return sorted((norm(i) for i in x), key=repr)
    return x


class _Timeout(BaseException):  # BaseException so `except Exception` can't swallow it
    pass


def _on_alarm(*_):
    raise _Timeout()


def run_call(fn, args, per_call=2.0):
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, per_call)
    try:
        return ["OK", repr(norm(fn(*copy.deepcopy(args))))]  # deepcopy: candidates may mutate inputs
    except _Timeout:
        return ["TIMEOUT"]
    except BaseException as e:
        return ["ERR", type(e).__name__]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def child(job_path, cid):
    """Run one candidate on all inputs in its own process."""
    job = json.load(open(job_path))
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")  # candidate prints must not corrupt our JSON
    try:
        ns = {"__name__": "candidate"}
        exec(compile(job["candidates"][cid], cid, "exec"), ns)
        fn = ns[job["entry_point"]]
    except BaseException as e:
        results = [["ERR", "LOAD:" + type(e).__name__]] * len(job["inputs"])
    else:
        results = []
        for args in job["inputs"]:
            a = run_call(fn, args)
            b = run_call(fn, args)
            results.append(a if a == b else ["FLAKY"])
    real_stdout.write(json.dumps(results))
    real_stdout.flush()
    os._exit(0)  # don't wait on threads the candidate may have started


def main(job_path):
    job = json.load(open(job_path))
    n = len(job["inputs"])
    limit = job.get("per_candidate_timeout", 20)
    out = {}
    for cid in job["candidates"]:
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--one", job_path, cid],
                capture_output=True, text=True, timeout=limit,
            )
            out[cid] = json.loads(p.stdout) if p.stdout else [["ERR", "CRASH"]] * n
        except subprocess.TimeoutExpired:
            out[cid] = [["TIMEOUT"]] * n
        except Exception:
            out[cid] = [["ERR", "HARNESS"]] * n
    print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        child(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
