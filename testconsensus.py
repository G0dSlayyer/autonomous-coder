"""Unit tests for the pure selection logic. No Docker, no LLM:  pytest -q"""
from consensus import choose, parse_examples, parse_inputs

OK = lambda v: ["OK", v]


def test_majority_cluster_wins():
    good = [OK("1"), OK("2")]
    sigs = [good] * 6 + [[OK("9"), OK("2")], [OK("8"), OK("2")], [["ERR", "X"]] * 2]
    r = choose(sigs, [])
    assert r["index"] == 0 and r["confidence"] == 6 / 9


def test_docstring_examples_filter_candidates():
    # candidates 0-2 agree with each other but fail the docstring example; 3-4 pass it
    wrong, right = [OK("1"), OK("5")], [OK("2"), OK("7")]
    sigs = [wrong, wrong, wrong, right, right]
    r = choose(sigs, expected=["2"])      # first input is the docstring example, expected '2'
    assert r["index"] in (3, 4) and not r["example_filter_fallback"]


def test_flaky_candidate_is_excluded_but_does_not_poison_inputs():
    good = [OK("1"), OK("2")]
    sigs = [[["FLAKY"], ["FLAKY"]], good, good]
    r = choose(sigs, [])
    assert r["index"] in (1, 2) and r["n_valid_inputs"] == 2


def test_tie_goes_to_earliest_sample():
    a, b = [OK("1")], [OK("2")]
    assert choose([a, a, b, b], [])["index"] == 0
    assert choose([b, b, a, a], [])["index"] == 0


def test_all_error_cluster_ignored_when_others_exist():
    err, ok = [["ERR", "X"]], [OK("1")]
    assert choose([err, err, err, ok], [])["index"] == 3


def test_parse_examples_reads_last_docstring_example():
    prompt = 'def f(a, b):\n    """ doc\n    >>> f(1, 2)\n    3\n    >>> f(0, 0)\n    0\n    """\n'
    assert parse_examples(prompt, "f") == [([1, 2], "3"), ([0, 0], "0")]


def test_parse_inputs_drops_garbage():
    assert parse_inputs('x [[1, 2], [1, 2], "bad", [3, 4]] y') == [[1, 2], [3, 4]]
    assert parse_inputs("no json here") == []


def test_agreement_is_measured_inside_the_filtered_pool():
    wrong, right = [OK("1"), OK("5")], [OK("2"), OK("7")]
    sigs = [wrong] * 8 + [right, right]
    r = choose(sigs, expected=["2"])            # only the 2 'right' candidates pass the example
    assert r["pool"] == [8, 9] and r["agreement"] == 1.0 and r["confidence"] == 0.2