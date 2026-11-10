"""
Eval harness for the autonomous coding agent (v3).

Methods:
  retry     : generate -> test -> feed failures back, up to --max-attempts   (default)
  consensus : sample --n candidates, run all on generated inputs in the sandbox,
              cluster by identical behavior, submit the biggest cluster

Feedback modes for `retry` (what the agent sees when it retries):
  self   : tests the agent wrote itself from the docstring  (honest, default)
  hidden : HumanEval's hidden tests                          (leaky upper bound)

In ALL modes the final score is computed on HumanEval's hidden tests.

Usage:
    python eval.py --limit 10
    python eval.py                                   # retry, self-test feedback, full set
    python eval.py --max-attempts 1                  # no-retry baseline
    python eval.py --feedback hidden                 # leaky ablation
    python eval.py --method consensus --n 10         # consensus
    python eval.py --model ollama:qwen2.5-coder:7b   # local model
    python eval.py --model llama-3.3-70b-versatile   # plain names = Groq models

Docker Desktop must be running.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import consensus
import inspect
import harness
from providers import DailyCap, extract_code, get_provider
from sandbox import classify, hidden_script, run_hidden_batch_kinds, run_in_sandbox

SOLVER_PROMPT = """You are an expert Python programmer. Complete the function the user gives you.
Respond with ONLY one Python code block containing the full function, including any imports
and helper functions it needs. No explanations."""

TESTER_PROMPT = """You are a meticulous test engineer. Given a Python function signature and docstring,
write 6-10 plain `assert` statements that test the function, based ONLY on the docstring.
Include the docstring examples, plus edge cases (empty input, single element, negatives, large values)
where relevant. Do NOT define or implement the function itself, and do not import it; assume it is
already defined. Use the docstring examples as-is, plus only trivial edge cases whose answer is obvious (e.g. empty
input). Never invent complex inputs whose expected output you would have to work out by hand. For floats, compare with abs(a - b) < 1e-6 instead of ==.
Respond with ONLY one Python code block of assert statements."""


def chat(provider, messages: list[dict]) -> tuple[str, int]:
    c = provider.chat(messages, temperature=0.2)
    return c.text, c.tokens

def example_tests(problem: dict) -> str:
    """Executable tests built from the docstring examples (parsed by code, not written by the LLM)."""
    entry = problem["entry_point"]
    lines = [inspect.getsource(harness.norm)]
    for args, want in consensus.parse_examples(problem["prompt"], entry):
        msg = f"{entry}(*{args!r}) expected {want}, got "
        lines.append(f"_r = repr(norm({entry}(*{args!r})))")
        lines.append(f"assert _r == {want!r}, {msg!r} + _r")
    return "\n".join(lines)

def generate_self_tests(problem: dict, provider) -> tuple[str, int]:
    text, tokens = chat(provider, [
        {"role": "system", "content": TESTER_PROMPT},
        {"role": "user", "content": f"```python\n{problem['prompt']}\n```"},
    ])
    return extract_code(text), tokens


# ---------------------------------------------------------------- method: retry

def solve(problem: dict, provider, max_attempts: int, feedback: str, fallback: str = "first") -> dict:
    start = time.time()
    tokens = 0
    self_tests = ""

    if feedback == "examples":  # real docstring examples as the tests; then same loop as "self"
        self_tests = example_tests(problem)
        feedback = "self"
    elif feedback == "self" and max_attempts > 1:
        self_tests, t = generate_self_tests(problem, provider)
        tokens += t

    messages = [
        {"role": "system", "content": SOLVER_PROMPT},
        {"role": "user", "content": f"```python\n{problem['prompt']}\n```"},
    ]

    first_hidden = False
    code = ""
    prev_code = ""
    feedback_pass = False
    attempts_used = 0

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        try:
            text, t = chat(provider, messages)
        except Exception as ex:  # small context window (e.g. 4K): stop retrying, keep what we have
            if attempt > 1 and "context_length_exceeded" in str(ex):
                break
            raise
        tokens += t
        code = extract_code(text)
        if attempt > 1 and code == prev_code:
            break  # model kept its code despite failing tests -> tests are suspect
        if fallback == "last":  # legacy: prev_code tracked the latest attempt
            prev_code = code    # (fallback "first": prev_code stays = attempt 1's code)

        if feedback == "self":
            feedback_pass, output = run_in_sandbox(f"{code}\n\n{self_tests}\n")
        else:
            feedback_pass, output = run_in_sandbox(hidden_script(code, problem))

        if attempt == 1:
            prev_code = code
            # Hidden-test score of the very first attempt = the no-retry baseline.
            first_hidden = (
                run_in_sandbox(hidden_script(code, problem))[0]
                if feedback == "self" else feedback_pass
            )

        if feedback_pass:
            break

        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": f"That failed. Output:\n\n{output}\n\nNote: the tests were auto-written and may themselves be wrong. If the failing assertion contradicts the docstring, trust the docstring and return the function unchanged. Otherwise return the full corrected function.",
        })

        if feedback == "self" and not feedback_pass:
            code = prev_code  # tests never passed -> they're suspect; fallback "first" = attempt 1's code
    # Final score is ALWAYS on the hidden tests.
    if feedback == "self":
        final_hidden, final_out = run_in_sandbox(hidden_script(code, problem))
    else:
        final_hidden, final_out = feedback_pass, output

    return {
        "task_id": problem["task_id"],
        "method": "retry",
        "fallback": fallback,
        "first_try_pass": first_hidden,
        "passed": final_hidden,
        "fail_kind": classify(final_hidden, final_out),
        "self_tests_passed": feedback_pass if feedback == "self" else None,
        "attempts_used": attempts_used,
        "tokens": tokens,
        "seconds": round(time.time() - start, 1),
        "last_output": output,
        "self_tests": self_tests,
    }


# ------------------------------------------------------------ method: consensus

def solve_consensus(problem: dict, provider, n: int, temperature: float, workers: int = 3) -> dict:
    start = time.time()
    r = consensus.solve(problem, provider, n=n, temperature=temperature, workers=workers)

    # ONE container scores all N candidates on the hidden tests. The winner is one of
    # them, so this gives: consensus pass, pass@1 (candidate 0), mean accuracy, oracle pass@N.
    kinds = run_hidden_batch_kinds(r["candidates"], problem)
    hidden = [k == "pass" for k in kinds]

    return {
        "task_id": problem["task_id"],
        "method": "consensus",
        "first_try_pass": hidden[0],                 # a single sample, no selection
        "passed": hidden[r["index"]],                # what consensus submits
        "cand_acc": sum(hidden) / len(hidden),       # expected pass@1 at this temperature
        "oracle": any(hidden),                       # upper bound: is ANY candidate right?
        "confidence": r["confidence"],               # winning cluster share of all N
        "agreement": r["agreement"],                 # winning cluster share of the filtered pool
        "pool_size": len(r["pool"]),
        # ABLATION: first candidate that passes the docstring examples, no clustering at all.
        # If this ~ equals consensus, the free filter does the work, not the clustering.
        "filter_only": hidden[min(r["pool"])],
        "pool_acc": sum(hidden[i] for i in r["pool"]) / len(r["pool"]),
        "index": r["index"],
        "cluster_sizes": r["cluster_sizes"],
        "n_valid_inputs": r["n_valid_inputs"],
        "example_filter_fallback": r["example_filter_fallback"],
        "hidden": hidden,
        "kinds": kinds,
        "attempts_used": n,
        "tokens": r["tokens"],
        "seconds": round(time.time() - start, 1),
    }


# ------------------------------------------------------------------------ driver

def safe_solve(problem, provider, args):
    try:
        if args.method == "consensus":
            return solve_consensus(problem, provider, args.n, args.temperature)
        return solve(problem, provider, args.max_attempts, args.feedback, args.fallback)
    except DailyCap:
        raise
    except Exception as e:  # e.g. API rate limit; record instead of crashing the run
        print(f"  ! {problem['task_id']} errored: {e}")
        return {"task_id": problem["task_id"], "error": str(e)}


def print_summary(args, ok, n, n_err, out_path):
    print("\n" + "=" * 50)
    print(f"Model:               {args.model}")
    print(f"Method:              {args.method}" +
          (f" (N={args.n}, T={args.temperature})" if args.method == "consensus"
           else f" (feedback: {args.feedback}, fallback: {args.fallback})"))
    print(f"Problems evaluated:  {n} ({n_err} errored)")

    if args.method == "consensus":
        print(f"pass@1 (sample 1):   {sum(r['first_try_pass'] for r in ok) / n:.1%}")
        print(f"mean candidate acc:  {sum(r['cand_acc'] for r in ok) / n:.1%}  (expected pass@1)")
        print(f"consensus pass:      {sum(r['passed'] for r in ok) / n:.1%}")
        print(f"oracle pass@{args.n}:      {sum(r['oracle'] for r in ok) / n:.1%}  (upper bound)")
        if all("filter_only" in r for r in ok):
            print(f"filter-only pass:    {sum(r['filter_only'] for r in ok) / n:.1%}  "
                  f"(first candidate passing the docstring examples, NO clustering)")
        print("calibration (agreement among filtered candidates -> actual accuracy):")
        for lo, hi in [(0.8, 1.01), (0.5, 0.8), (0.0, 0.5)]:
            b = [r for r in ok if lo <= r.get("agreement", r["confidence"]) < hi]
            if b:
                print(f"  conf {lo:.1f}-{min(hi, 1.0):.1f}: {sum(r['passed'] for r in b) / len(b):.1%} correct (n={len(b)})")
    else:
        print(f"pass@1 (hidden):     {sum(r['first_try_pass'] for r in ok) / n:.1%}")
        print(f"final pass (hidden): {sum(r['passed'] for r in ok) / n:.1%}  (up to {args.max_attempts} attempts)")
        if args.feedback == "self":
            print(f"self-tests passed:   {sum(bool(r['self_tests_passed']) for r in ok) / n:.1%}")
        print(f"avg attempts:        {sum(r['attempts_used'] for r in ok) / n:.2f}")

    print(f"avg tokens/problem:  {sum(r['tokens'] for r in ok) / n:.0f}")
    print(f"avg seconds/problem: {sum(r['seconds'] for r in ok) / n:.1f}")
    print(f"Saved to {out_path}")


def main():
    from datasets import load_dataset  # imported here so the other modules work without it

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b",
                    help="Groq model name, or groq:<model> / ollama:<model>")
    ap.add_argument("--method", choices=["retry", "consensus"], default="retry")
    ap.add_argument("--n", type=int, default=10, help="candidates for consensus")
    ap.add_argument("--temperature", type=float, default=0.8, help="sampling temperature for consensus")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--feedback", choices=["self", "hidden", "examples"], default="self")
    ap.add_argument("--fallback", choices=["first", "last"], default="first",
                    help="retry, self-feedback: if the self-tests never pass, submit attempt 1 (first, default) "
                         "or the final attempt (last = old behavior)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    spec = args.model if args.model.startswith(("groq:", "ollama:")) else f"groq:{args.model}"
    provider = get_provider(spec)

    problems = list(load_dataset("openai/openai_humaneval", split="test"))
    problems = problems[args.start:]
    if args.limit:
        problems = problems[: args.limit]

    print(f"Model: {args.model} | method: {args.method} | problems: {len(problems)} | " +
          (f"N: {args.n}" if args.method == "consensus"
           else f"max attempts: {args.max_attempts} | feedback: {args.feedback}"))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(safe_solve, p, provider, args) for p in problems]
        for i, fut in enumerate(futures, 1):
            try:
                r = fut.result()
            except DailyCap as e:
                print(f"\nDaily cap hit ({e}). Stopping; nothing saved for the rest.")
                for f in futures:
                    f.cancel()
                break
            results.append(r)
            if "error" not in r:
                status = "PASS" if r["passed"] else "FAIL"
                extra = (f"conf {r['confidence']:.0%}, oracle {r['oracle']}"
                         if args.method == "consensus" else f"{r['attempts_used']} attempts")
                print(f"[{i}/{len(problems)}] {r['task_id']}: {status} ({extra})")

    if not any("error" not in r for r in results):
        print("Nothing succeeded; not saving. See the errors above (bad model name? daily cap?).")
        return
    os.makedirs("results", exist_ok=True)
    tag = re.sub(r"[/:]", "_", args.model)  # ':' is illegal in Windows filenames
    if args.method == "consensus":
        name = f"{tag}_consensus_N{args.n}_s{args.start}_n{len(problems)}.jsonl"
    else:
        fb_tag = "first" if (args.fallback == "first" and args.feedback in ("self", "examples")) else ""
        name = f"{tag}_{args.feedback}_k{args.max_attempts}{fb_tag}_s{args.start}_n{len(problems)}.jsonl"
    out_path = os.path.join("results", name)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if "error" not in r]
    print_summary(args, ok, len(ok), len(results) - len(ok), out_path)


if __name__ == "__main__":
    main()