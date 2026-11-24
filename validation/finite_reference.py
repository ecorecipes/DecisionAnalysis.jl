"""Independent reference values for the finite core of DecisionAnalysis.jl.

Run from the package root with the virtual environment of validation/README.md:

    validation/.venv/bin/python validation/finite_reference.py

It writes test/fixtures/validation/finite_reference.json, which
test/test_validation.jl reads; the Julia test suite needs neither Python nor R.

Two kinds of check, on the three worked examples of
test/fixtures/worked_examples.json and on seeded random problems:

* the textbook payoff-table criteria (maximin, maximax, Hurwicz, Laplace, minimax
  regret, expected utility, expected regret, EVPI, lower quantile), written here from
  the definitions in exact rational arithmetic (`fractions.Fraction`), with no code
  shared with the package;
* everything the package decides by linear programme, solved here by
  `scipy.optimize.linprog` (HiGHS) from formulations written independently:
  conditional value at risk (the Rockafellar-Uryasev programme), lower and upper
  expectations over a credal set and Gamma-maximin, E-admissibility (a feasibility
  programme, not the package's slack-maximising one), maximality and interval
  dominance (from the generating points), the garbling test (a feasibility programme),
  Le Cam's deficiency and the upper envelope of EVPI over a credal set.

Layouts as in the package: payoff[a][state]; likelihood[signal][state], columns
summing to one; a garbling G[new signal][old signal], column-stochastic.
"""

import json
import platform
import random
from fractions import Fraction
from pathlib import Path

import numpy
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test" / "fixtures" / "validation" / "finite_reference.json"
HURWICZ = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(7, 10), Fraction(1)]
LEVELS = [Fraction(1, 20), Fraction(1, 4), Fraction(1, 2), Fraction(9, 10), Fraction(1)]


def frac(text):
    return Fraction(text)


def text(x):
    return str(Fraction(x))


def floats(rows):
    return [[float(x) for x in row] for row in rows]


# Textbook criteria, exact -------------------------------------------------------------


def criteria(payoff):
    """State-wise criteria of a payoff table of utilities; no probabilities."""
    states = range(len(payoff[0]))
    best_in_state = [max(row[s] for row in payoff) for s in states]
    regret = [[best_in_state[s] - row[s] for s in states] for row in payoff]
    return {
        "maximin": [min(row) for row in payoff],
        "maximax": [max(row) for row in payoff],
        "laplace": [sum(row) / len(row) for row in payoff],
        # Hurwicz with alpha the weight on the BEST outcome (index of optimism).
        "hurwicz": [[a * max(row) + (1 - a) * min(row) for row in payoff] for a in HURWICZ],
        "regret": regret,
        "max_regret": [max(row) for row in regret],
    }


def expectations(payoff, prior):
    return [sum(p * q for p, q in zip(prior, row)) for row in payoff]


def evpi(payoff, prior):
    states = range(len(prior))
    perfect = sum(prior[s] * max(row[s] for row in payoff) for s in states)
    return perfect - max(expectations(payoff, prior))


def lower_quantile(row, prior, level):
    """Smallest payoff x with P(payoff <= x) >= level."""
    for x in sorted(set(row)):
        if sum(p for p, q in zip(prior, row) if q <= x) >= level:
            return x
    return max(row)


def value_with_information(payoff, prior, likelihood):
    total = Fraction(0)
    for signal in likelihood:
        total += max(sum(p * l * q for p, l, q in zip(prior, signal, row)) for row in payoff)
    return total


# Linear programmes, by scipy ----------------------------------------------------------


def solve(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")


def cvar(row, prior, level):
    """Rockafellar and Uryasev: max_t t - E[(t - X)^+] / level, for a utility X."""
    n = len(row)
    # Variables t, u_1..u_n with u_i >= t - x_i, u_i >= 0; maximise t - sum p_i u_i / level.
    c = [-1.0] + [float(p) / float(level) for p in prior]
    A = [[1.0] + [-1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    b = [float(x) for x in row]
    res = solve(c, A_ub=A, b_ub=b, bounds=[(None, None)] + [(0, None)] * n)
    assert res.status == 0
    return -res.fun


def hull_extreme(values, sense):
    """min or max of a linear function over the hull of the generating points."""
    k = len(values)
    res = solve([sense * float(v) for v in values], A_eq=[[1.0] * k], b_eq=[1.0],
                bounds=[(0, None)] * k)
    assert res.status == 0
    return sense * res.fun


def e_admissible(E, a):
    """Is there a mixture of the generating points under which action a is optimal?"""
    n, k = len(E), len(E[0])
    A = [[float(E[b][j] - E[a][j]) for j in range(k)] for b in range(n) if b != a]
    res = solve([0.0] * k, A_ub=A or None, b_ub=[0.0] * len(A) or None, A_eq=[[1.0] * k],
                b_eq=[1.0], bounds=[(0, None)] * k)
    return res.status == 0


def maximal(E, a):
    """No other action is strictly better at every generating point."""
    return not any(all(E[b][j] > E[a][j] for j in range(len(E[0])))
                   for b in range(len(E)) if b != a)


def evpi_upper(payoff, points):
    """max over the hull of the single-prior EVPI: maximise sum l_k m_k - w."""
    E = [expectations(payoff, p) for p in points]            # E[k][a]
    states = range(len(points[0]))
    perfect = [sum(p[s] * max(row[s] for row in payoff) for s in states) for p in points]
    k, n = len(points), len(payoff)
    c = [-float(m) for m in perfect] + [1.0]
    A = [[float(E[j][a]) for j in range(k)] + [-1.0] for a in range(n)]
    res = solve(c, A_ub=A, b_ub=[0.0] * n, A_eq=[[1.0] * k + [0.0]], b_eq=[1.0],
                bounds=[(0, None)] * k + [(None, None)])
    assert res.status == 0
    return -res.fun


def garbling_variables(nt, nz):
    return [(zt, z) for zt in range(nt) for z in range(nz)]


def is_garbling(target, source):
    """Is `target` = G `source` for a column-stochastic G? A feasibility programme."""
    nt, nz, S = len(target), len(source), len(source[0])
    index = {v: i for i, v in enumerate(garbling_variables(nt, nz))}
    A_eq, b_eq = [], []
    for z in range(nz):
        row = [0.0] * len(index)
        for zt in range(nt):
            row[index[(zt, z)]] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    for zt in range(nt):
        for s in range(S):
            row = [0.0] * len(index)
            for z in range(nz):
                row[index[(zt, z)]] = float(source[z][s])
            A_eq.append(row)
            b_eq.append(float(target[zt][s]))
    res = solve([0.0] * len(index), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * len(index))
    return res.status == 0


def deficiency(source, target):
    """min_G max_state TV(G source[., state], target[., state]); TV is half the l1 norm."""
    nt, nz, S = len(target), len(source), len(source[0])
    g = {v: i for i, v in enumerate(garbling_variables(nt, nz))}
    u = {(zt, s): len(g) + zt * S + s for zt in range(nt) for s in range(S)}
    d = len(g) + nt * S
    size = d + 1
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for z in range(nz):
        row = [0.0] * size
        for zt in range(nt):
            row[g[(zt, z)]] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    for zt in range(nt):
        for s in range(S):
            for sign in (1.0, -1.0):
                row = [0.0] * size
                for z in range(nz):
                    row[g[(zt, z)]] = sign * float(source[z][s])
                row[u[(zt, s)]] = -1.0
                A_ub.append(row)
                b_ub.append(sign * float(target[zt][s]))
    for s in range(S):
        row = [0.0] * size
        for zt in range(nt):
            row[u[(zt, s)]] = 1.0
        row[d] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    c = [0.0] * size
    c[d] = 1.0
    res = solve(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * size)
    assert res.status == 0
    return res.fun / 2


# Problems -----------------------------------------------------------------------------


def random_probabilities(rng, n, denominator=20):
    cuts = sorted(rng.randint(0, denominator) for _ in range(n - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [denominator])]
    return [Fraction(p, denominator) for p in parts]


def random_likelihood(rng, signals, states):
    columns = [random_probabilities(rng, signals, 10) for _ in range(states)]
    return [[columns[s][z] for s in range(states)] for z in range(signals)]


def multiply(G, L):
    return [[sum(G[i][k] * L[k][j] for k in range(len(L))) for j in range(len(L[0]))]
            for i in range(len(G))]


def analyse(name, payoff, prior, credal, experiments):
    out = {"name": name, "payoff": [[text(x) for x in row] for row in payoff],
           "prior": [text(p) for p in prior],
           "credal": [[text(p) for p in point] for point in credal]}
    crit = criteria(payoff)
    out["maximin"] = [text(x) for x in crit["maximin"]]
    out["maximax"] = [text(x) for x in crit["maximax"]]
    out["laplace"] = [text(x) for x in crit["laplace"]]
    out["hurwicz_optimism"] = [text(a) for a in HURWICZ]
    out["hurwicz"] = [[text(x) for x in scores] for scores in crit["hurwicz"]]
    out["max_regret"] = [text(x) for x in crit["max_regret"]]
    regret = crit["regret"]
    out["expected_utility"] = [text(x) for x in expectations(payoff, prior)]
    out["expected_regret"] = [text(x) for x in expectations(regret, prior)]
    out["evpi"] = text(evpi(payoff, prior))
    out["levels"] = [text(a) for a in LEVELS]
    out["quantile"] = [[text(lower_quantile(row, prior, a)) for row in payoff] for a in LEVELS]
    out["cvar_linprog"] = [[cvar(row, prior, a) for row in payoff] for a in LEVELS]
    # The credal set: E[a][k], the expected utility of action a at generating point k.
    E = [[sum(p * q for p, q in zip(point, row)) for point in credal] for row in payoff]
    lower = [hull_extreme(row, 1.0) for row in E]
    upper = [hull_extreme(row, -1.0) for row in E]
    out["lower_expectation_linprog"] = lower
    out["upper_expectation_linprog"] = upper
    actions = range(len(payoff))
    out["e_admissible_linprog"] = [e_admissible(E, a) for a in actions]
    out["maximal"] = [maximal(E, a) for a in actions]
    out["interval_dominance"] = [max(E[a]) >= max(min(row) for row in E) for a in actions]
    out["evpi_lower"] = text(min(evpi(payoff, point) for point in credal))
    out["evpi_upper_linprog"] = evpi_upper(payoff, credal)
    out["experiments"] = []
    for label, L in experiments:
        out["experiments"].append({
            "name": label, "likelihood": [[text(x) for x in row] for row in L],
            "evsi": text(value_with_information(payoff, prior, L) - max(expectations(payoff, prior)))})
    pairs = []
    for i, (name_i, Li) in enumerate(experiments):
        for j, (name_j, Lj) in enumerate(experiments):
            if i != j:
                # Is experiment i a garbling of experiment j, and how deficient is j
                # with respect to i?
                pairs.append({"garbled": name_i, "of": name_j,
                              "is_garbling_linprog": is_garbling(Li, Lj),
                              "deficiency_linprog": deficiency(Lj, Li)})
    out["pairs"] = pairs
    return out


def worked_examples():
    data = json.loads((ROOT / "test" / "fixtures" / "worked_examples.json").read_text())
    problems = []
    for ex in data["examples"]:
        payoff = [[frac(x) for x in row] for row in ex["payoff"]]
        prior = [frac(x) for x in ex["prior"]]
        credal = [[frac(x) for x in point] for point in ex.get("credal", [ex["prior"]])]
        experiments = [(label, [[frac(x) for x in row] for row in e["likelihood"]])
                       for label, e in ex.get("experiments", {}).items()]
        problems.append(analyse(ex["name"], payoff, prior, credal, experiments))
    return problems


def random_problems(count=30, seed=20261123):
    rng = random.Random(seed)
    problems = []
    for i in range(count):
        m, n = rng.randint(2, 5), rng.randint(2, 5)
        payoff = [[Fraction(rng.randint(-12, 12)) for _ in range(n)] for _ in range(m)]
        prior = random_probabilities(rng, n)
        credal = [random_probabilities(rng, n) for _ in range(rng.randint(1, 4))]
        first = random_likelihood(rng, rng.randint(2, 3), n)
        # A garbling of the first experiment, and an unrelated one.
        G = random_likelihood(rng, rng.randint(2, 3), len(first))
        experiments = [("e", first), ("garbled", multiply(G, first)),
                       ("other", random_likelihood(rng, rng.randint(2, 3), n))]
        problems.append(analyse(f"random{i + 1}", payoff, prior, credal, experiments))
    return problems


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "provenance": {"python": platform.python_version(), "numpy": numpy.__version__,
                       "scipy": scipy.__version__, "solver": "scipy.optimize.linprog, method='highs'",
                       "script": "validation/finite_reference.py", "seed": 20261123},
        "conventions": {
            "hurwicz": "alpha weights the best outcome (index of optimism)",
            "cvar": "expected utility in the worst `level` share of the probability mass",
            "quantile": "smallest payoff x with P(payoff <= x) >= level",
            "deficiency": "total variation is half the l1 distance",
            "linprog": "entries ending in _linprog are floating point; the rest are exact"},
        "problems": worked_examples() + random_problems()}
    OUT.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(document['problems'])} problems")


if __name__ == "__main__":
    main()
