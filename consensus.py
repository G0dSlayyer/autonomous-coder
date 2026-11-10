"""Execution-based consensus: sample N candidates, run them all on generated
inputs in the sandbox, cluster by identical behavior, pick the biggest cluster.

The LLM only writes candidates and inputs. Verification is plain code.
`choose()` is pure (no Docker, no LLM) so you can unit-test it.
"""
import ast
import doctest
import json
import re
import time
from collections import defaultdict

from harness import norm
from providers import extract_code
import sandbox

SOLVE_SYSTEM = "You are an expert Python programmer."
SOLVE_PROMPT = """Complete the Python code below so that it satisfies its docstring.

{prompt}

Reply with ONE ```python block containing the complete code: imports, any helper
functions already defined above, and the finished function. No explanation."""

INPUT_PROMPT = """Below is a Python function specification. Write 25 diverse test INPUTS
for the function `{entry}`. Do NOT write expected outputs.

Return ONLY a JSON list. Each item is the list of positional arguments for one call.
Cover edge cases: empty values, single elements, zero, negatives, duplicates, large
values. Every input must be valid according to the specification.

{prompt}"""


# ---------- parsing ----------

def parse_inputs(text, limit=30):
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b < 0:
        return []
    try:
        raw = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return []
    seen, out = set(), []
    for item in raw:
        if not isinstance(item, list):
            continue
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:limit]


def parse_examples(prompt, entry):
    """Docstring examples (visible to the agent, so legitimate): [(args, expected_repr)]"""
    out = []
    text = prompt
    try:  # use just the docstring so the closing quotes don't leak into `want`
        for node in ast.walk(ast.parse(prompt)):
            if isinstance(node, ast.FunctionDef) and node.name == entry:
                text = ast.get_docstring(node) or prompt
    except SyntaxError:
        pass
    for ex in doctest.DocTestParser().get_examples(text):
        try:
            node = ast.parse(ex.source.strip(), mode="eval").body
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == entry and not node.keywords):
                continue
            args = [ast.literal_eval(a) for a in node.args]
            if json.loads(json.dumps(args)) != args:  # e.g. tuples would change in JSON
                continue
            want = ast.literal_eval(ex.want.strip())
            out.append((args, repr(norm(want))))
        except Exception:
            continue
    return out


# ---------- the core: pure clustering logic ----------

def choose(sigs, expected):
    """sigs[i][j] = [status, value?] for candidate i on input j.
    The first len(expected) inputs are docstring examples with known expected reprs.
    Returns the winner index plus diagnostics."""
    n, n_ex = len(sigs), len(expected)

    # 1. filter: must pass every docstring example and be deterministic
    ex_pass = [all(sigs[i][j] == ["OK", expected[j]] for j in range(n_ex)) for i in range(n)]
    flaky = [any(r[0] == "FLAKY" for r in sigs[i]) for i in range(n)]  # nondeterministic = bad candidate
    pool = [i for i in range(n) if ex_pass[i] and not flaky[i]]
    fallback = not pool
    if fallback:
        pool = [i for i in range(n) if not flaky[i]] or list(range(n))

    # 2. keep only informative inputs: at least one pooled candidate ran OK
    n_in = len(sigs[0]) if sigs else 0
    cols = []
    for j in range(n_ex, n_in):
        if not any(sigs[i][j][0] == "OK" for i in pool):
            continue
        cols.append(j)

    # 3. cluster by behavior signature
    groups = defaultdict(list)
    for i in pool:
        groups[tuple(tuple(sigs[i][j]) for j in cols)].append(i)

    # 4. ignore clusters that produced no successful output at all, if others exist
    useful = {s: m for s, m in groups.items() if any(r[0] == "OK" for r in s)} or groups

    # 5. biggest cluster wins; ties go to the cluster containing the earliest sample
    best = max(useful.values(), key=lambda m: (len(m), -min(m)))
    return {
        "index": min(best),
        "confidence": len(best) / n,          # winning cluster share of ALL n candidates
        "agreement": len(best) / len(pool),   # share among candidates that passed the docstring examples
        "pool": pool,                         # indices that passed the examples (the free filter)
        "cluster_sizes": sorted((len(m) for m in groups.values()), reverse=True),
        "n_valid_inputs": len(cols),
        "pool_size": len(pool),
        "example_filter_fallback": fallback,
    }


# ---------- orchestration (LLM + sandbox) ----------

def solve(problem, provider, n=10, temperature=0.8, workers=4):
    t0 = time.time()
    prompt, entry = problem["prompt"], problem["entry_point"]
    header = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))

    comps = provider.generate(SOLVE_PROMPT.format(prompt=prompt), system=SOLVE_SYSTEM,
                              temperature=temperature, n=n, workers=workers)
    codes = [header + "\n" + extract_code(c.text) for c in comps]
    tokens = sum(c.tokens for c in comps)

    ic = provider.generate(INPUT_PROMPT.format(entry=entry, prompt=prompt), temperature=0.3)[0]
    tokens += ic.tokens

    examples = parse_examples(prompt, entry)
    inputs = [a for a, _ in examples] + parse_inputs(ic.text)
    expected = [w for _, w in examples]

    job = {"entry_point": entry,
           "candidates": {f"c{i}": c for i, c in enumerate(codes)},
           "inputs": inputs}
    res = sandbox.run_job(job)
    sigs = [res[f"c{i}"] for i in range(len(codes))]

    out = choose(sigs, expected)
    out.update(code=codes[out["index"]], candidates=codes, tokens=tokens,
               seconds=time.time() - t0)
    return out