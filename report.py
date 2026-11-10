"""
Aggregate eval results into a README-ready table and charts.

Usage:
    pip install matplotlib
    python report.py

Reads results/*.jsonl produced by eval.py:
    <model>_<self|hidden>_k<K>_s<start>_n<limit>.jsonl     (--method retry)
    <model>_consensus_N<N>_s<start>_n<limit>.jsonl         (--method consensus)
Merges chunks per (model, method, settings), de-duplicates by task_id, and writes
results/summary.md and results/summary.png.

Chart notes: bars are computed on the problems EVERY run of that model has in common,
so methods are compared like for like. Error bars are 95% Wilson intervals.
"""
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

RETRY = re.compile(r"^(?P<model>.+)_(?P<fb>self|hidden|examples)_k(?P<k>\d+)(?P<f1>first)?_s\d+_n\d+\.jsonl$")
CONS = re.compile(r"^(?P<model>.+)_consensus_N(?P<n>\d+)_s\d+_n\d+\.jsonl$")


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt(passes: int, n: int) -> str:
    lo, hi = wilson(passes, n)
    return f"{passes / n:.1%} ({lo:.0%}-{hi:.0%})"


def load() -> dict:
    """(model, kind, param) -> {task_id: record};  kind is 'retry' or 'consensus'."""
    groups = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join("results", "*.jsonl"))):
        name = os.path.basename(path)
        if m := RETRY.match(name):
            key = (m["model"], "retry", (m["fb"], int(m["k"]), bool(m["f1"])))
        elif m := CONS.match(name):
            key = (m["model"], "consensus", (int(m["n"]),))
        else:
            continue  # skips old/unchunked files
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    groups[key][r["task_id"]] = r
    return groups


def label(kind, param) -> str:
    if kind == "retry":
        fb, k, f1 = param
        return f"retry ({fb}, k={k}{', fallback to attempt 1' if f1 else ''})"
    return f"consensus (N={param[0]})"


def order(item):
    (model, kind, param), _ = item
    return (model, 0 if kind == "retry" else 1, param)

def group(kind: str) -> str:
    return kind.split(":")[0]  # runtime_error:NameError -> runtime_error


def taxonomy_notes(items) -> list:
    out = []
    for (model, kind, param), recs in items:
        rs = list(recs.values())
        lines = []
        if kind == "consensus":
            b = Counter()
            for r in rs:
                if r["passed"]:
                    b["correct"] += 1
                elif r["oracle"]:
                    b["selection miss (a correct candidate existed)"] += 1
                elif "kinds" in r:
                    top = Counter(group(k) for k in r["kinds"]).most_common(1)[0][0]
                    b[f"no correct candidate, mostly {top}"] += 1
                else:
                    b["no correct candidate"] += 1
            lines.append("final answers: " + ", ".join(f"{k} {v}" for k, v in b.most_common()))
            if all("kinds" in r for r in rs):
                c = Counter(group(k) for r in rs for k in r["kinds"])
                total = sum(c.values())
                lines.append("every candidate: " + ", ".join(f"{k} {v / total:.0%}" for k, v in c.most_common()))
                errs = Counter(k for r in rs for k in r["kinds"] if k.startswith("runtime_error"))
                if errs:
                    lines.append("runtime errors: " + ", ".join(f"{k.split(':')[1]} {v}" for k, v in errs.most_common(5)))
        else:
            if all("fail_kind" in r for r in rs):
                c = Counter(group(r["fail_kind"]) for r in rs)
                lines.append("final answers: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
            if all(r.get("self_tests_passed") is not None for r in rs):
                t = Counter()
                for r in rs:
                    st, hp = r["self_tests_passed"], r["passed"]
                    t["tests ok, code ok" if st and hp else
                      "tests too weak (accepted wrong code)" if st else
                      "tests wrong (rejected correct code)" if hp else
                      "code wrong, tests caught it"] += 1
                lines.append("own tests vs hidden tests: " + ", ".join(f"{k} {v}" for k, v in t.most_common()))
        if lines:
            out.append(f"- **{model}**, {label(kind, param)}")
            out += [f"  - {l}" for l in lines]
    return ["", "Failure taxonomy:", *out] if out else []

def main():
    groups = load()
    if not groups:
        print("No result files found in results/.")
        return
    items = sorted(groups.items(), key=order)

    # ---------------- main table (each run on all of its own problems) ----------------
    ablations = []
    lines = [
        "| Model | Method | Problems | pass@1 (95% CI) | Final (95% CI) | Gain | Oracle pass@N | Mean sample acc | Avg tokens | Avg sec |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, kind, param), recs in items:
        rs = list(recs.values())
        n = len(rs)
        p1 = sum(r["first_try_pass"] for r in rs)
        pk = sum(r["passed"] for r in rs)
        oracle = fmt(sum(r["oracle"] for r in rs), n) if kind == "consensus" else "-"
        mean_acc = f"{sum(r['cand_acc'] for r in rs) / n:.1%}" if kind == "consensus" else "-"
        # consensus: compare against the AVERAGE sample (stable), not the single first sample
        gain = (pk / n - sum(r["cand_acc"] for r in rs) / n) if kind == "consensus" else (pk - p1) / n
        if kind == "consensus" and all("filter_only" in r for r in rs):
            fo = sum(r["filter_only"] for r in rs)
            ablations.append(f"- {model}, {label(kind, param)}: filter-only (first candidate passing the "
                             f"docstring examples, no clustering) = {fmt(fo, n)} vs full consensus = {fmt(pk, n)}.")
        lines.append(
            f"| {model} | {label(kind, param)} | {n} | {fmt(p1, n)} | {fmt(pk, n)} | "
            f"{gain:+.1%} | {oracle} | {mean_acc} | "
            f"{sum(r['tokens'] for r in rs) / n:.0f} | {sum(r['seconds'] for r in rs) / n:.1f} |"
        )
    notes = [
        "",
        "Gain: retry = final - pass@1; consensus = final - mean sample acc. "
        "pass@1 = first attempt (retry) or one sampled candidate at T>0 (consensus). "
        "Mean sample acc = average accuracy over all N candidates (expected pass@1 at that temperature). "
        "Oracle = at least one of N candidates passes (upper bound for any selection method).",
    ]

    # ---------------- paired comparison on common problems, per model ----------------
    by_model = defaultdict(list)
    for (model, kind, param), recs in items:
        by_model[model].append((kind, param, recs))

    series_by_model, paired = {}, []
    for model, gs in by_model.items():
        common = set.intersection(*(set(recs) for _, _, recs in gs))
        if not common:
            continue
        n = len(common)
        pick = lambda recs, key: sum(bool(recs[t][key]) for t in common)
        series = []
        retry_gs = sorted((g for g in gs if g[0] == "retry"), key=lambda g: not g[1][2])  # fixed runs first
        if retry_gs:   # true no-retry baseline: first attempt of a retry run
            series.append(("pass@1", pick(retry_gs[0][2], "first_try_pass"), n))
        else:
            series.append(("pass@1 (1 sample)", pick(gs[0][2], "first_try_pass"), n))
        for kind, param, recs in gs:
            if kind == "retry" and param[1] == 1:
                continue  # identical to the pass@1 bar
            series.append((label(kind, param), pick(recs, "passed"), n))
            if kind == "consensus":
                series.append((f"oracle pass@{param[0]}", pick(recs, "oracle"), n))
        series_by_model[model] = (series, n)

        # discordant pairs: where do consensus and the best retry run disagree?
        cons = [g for g in gs if g[0] == "consensus"]
        if cons and retry_gs:
            best_retry = max(retry_gs, key=lambda g: (g[1][2], g[1][1]))
            for ck, cp, crecs in cons:
                wins = sum(crecs[t]["passed"] and not best_retry[2][t]["passed"] for t in common)
                loses = sum(best_retry[2][t]["passed"] and not crecs[t]["passed"] for t in common)
                paired.append(f"- **{model}**, {label(ck, cp)} vs {label('retry', best_retry[1])} "
                              f"on {n} common problems: consensus fixes {wins}, breaks {loses}.")
    if paired:
        notes += ["", "Paired comparison (same problems):", *paired]
    if ablations:
        notes += ["", "Ablation (does clustering add anything beyond the free docstring-example filter?):", *ablations]
        notes += taxonomy_notes(items)
    out = "\n".join(lines + notes)
    print(out)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "summary.md"), "w", encoding="utf-8") as f:
        f.write(out + "\n")

    # ---------------- charts ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(pip install matplotlib to also get a chart)")
        return

    cons_groups = [(m, p, recs) for (m, k, p), recs in items if k == "consensus"]
    ncols = 2 if cons_groups else 1
    n_bars = sum(len(s) + 1 for s, _ in series_by_model.values())
    fig, axes = plt.subplots(1, ncols, squeeze=False,
                             figsize=(max(6, 0.9 * n_bars) + (5 if cons_groups else 0), 5.4))
    ax = axes[0][0]

    x0, ticks, ticklabels, colors, labeled = 0, [], [], {}, set()
    for model, (series, n) in series_by_model.items():
        for j, (name, p, nn) in enumerate(series):
            colors.setdefault(name, f"C{len(colors)}")
            v, (lo, hi) = p / nn * 100, tuple(c * 100 for c in wilson(p, nn))
            ax.bar(x0 + j, v, 0.8, color=colors[name], yerr=[[v - lo], [hi - v]], capsize=3,
                   label=None if name in labeled else name)
            labeled.add(name)
        ticks.append(x0 + (len(series) - 1) / 2)
        ticklabels.append(f"{model.replace('openai_', '')}\n(n={n})")
        x0 += len(series) + 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=9)
    ax.set_ylabel("HumanEval pass rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("pass@1 vs retry loop vs consensus", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)

    if cons_groups:
        ax2 = axes[0][1]
        bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        for model, param, recs in cons_groups:
            xs, ys, ns = [], [], []
            for lo, hi in bins:
                b = [r for r in recs.values() if lo <= r.get("agreement", r["confidence"]) < hi]
                if b:
                    xs.append((lo + min(hi, 1.0)) / 2)
                    ys.append(sum(r["passed"] for r in b) / len(b) * 100)
                    ns.append(len(b))
            ax2.plot(xs, ys, "o-", label=f"{model.replace('openai_', '')} N={param[0]}")
            for x, y, c in zip(xs, ys, ns):
                ax2.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(0, 6),
                             ha="center", fontsize=7)
        ax2.set_xlabel("Agreement (winning cluster share of filtered candidates)")
        ax2.set_ylabel("Actually correct (%)")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 108)
        ax2.set_title("Calibration: does agreement predict correctness?", fontsize=11)
        ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join("results", "summary.png"), dpi=150)
    print("\nSaved results/summary.md and results/summary.png")


if __name__ == "__main__":
    main()