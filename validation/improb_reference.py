# -*- coding: utf-8 -*-
"""Imprecise-probability decision rules computed by the Python package improb.

Python 2.7 only (improb 0.1.1 does not run on Python 3). Run from the package root
with the environment of validation/README.md:

    validation/.venv27/bin/python validation/improb_reference.py

It writes test/fixtures/validation/improb_reference.json, which
test/test_validation_improb.jl reads; the Julia test suite needs no Python.

Inputs. The payoff tables and credal sets are read from
test/fixtures/validation/finite_reference.json, so that this script and
validation/finite_reference.py work on identical problems (the three worked examples
and the seeded random problems; exact rationals as strings).

The credal set. DecisionAnalysis.jl holds a credal set as a list of generating points.
improb holds a lower prevision as a set of bounds on expectations (half-spaces). The
set is handed over as `LowPoly(pspace, credalset=points, number_type='fraction')`:
improb passes the points to cddlib as a V-representation, takes the H-representation
of their convex hull, and stores one bound per facet. Every quantity below is then
computed by improb from those facets, in exact arithmetic (`fractions.Fraction`, cddlib
with GMP rationals), not from the points. As a check of the hand-over the script turns
the facets back into extreme points (`LowPoly.get_credal_set`) and asserts that they are
among the points given; they are written to the fixture.

What improb computes:

* lower and upper expectations: `LowPoly.get_lower`, `LowPoly.get_upper` (natural
  extension, a linear programme over the facets);
* Gamma-maximin: `improb.decision.opt.OptLowPrevMaxMin` (all tied actions);
* maximality: `OptLowPrevMax` (b beats a when the lower prevision of b - a is > 0);
* interval dominance: `OptLowPrevMaxInterval` (b beats a when lower(b) > upper(a)).

E-admissibility. improb has no operator for it (neither release 0.1.1 nor the master
branch of github.com/mcmtroffaes/improb: `OptAdmissible` is pointwise dominance between
gambles, a different notion). It is therefore composed here from improb's primitives:
action a is E-admissible exactly when the credal set, with the extra assessments
"lower prevision of (a - b) >= 0" for every other action b, is not empty, that is when
that lower prevision avoids sure loss (`LowPoly.set_lower`, then
`LowPoly.is_avoiding_sure_loss`, the algorithm of Walley, Pelessoni and Vicig 2004).
The script also asks cddlib for the extreme points of the constrained set and asserts
that the two answers agree. The arithmetic is improb's; the reduction is this script's.
"""

from __future__ import division, print_function

import io
import json
import os
import platform
from fractions import Fraction

import cdd
import improb
from improb.decision.opt import OptLowPrevMax, OptLowPrevMaxInterval, OptLowPrevMaxMin
from improb.lowprev.lowpoly import LowPoly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "test", "fixtures", "validation", "finite_reference.json")
OUT = os.path.join(ROOT, "test", "fixtures", "validation", "improb_reference.json")


def text(x):
    return str(Fraction(x))


def credal_lower_prevision(points, states):
    """improb's lower prevision of the convex hull of `points` (lists of strings)."""
    return LowPoly(pspace=states, credalset=[[str(p) for p in point] for point in points],
                   number_type="fraction")


def chosen(opt, gambles):
    """Indices of the gambles that the optimality operator keeps (by identity)."""
    kept = list(opt(gambles))
    return [i for i, g in enumerate(gambles) if any(g is k for k in kept)]


def e_admissible(points, states, payoff, a):
    """Is the credal set, cut down to the priors under which `a` is optimal, non-empty?"""
    cut = credal_lower_prevision(points, states)
    for b, row in enumerate(payoff):
        if b != a:
            difference = [x - y for x, y in zip(payoff[a], row)]
            if any(difference):
                cut.set_lower(difference, 0)
            # An identical row constrains nothing.
    answer = cut.is_avoiding_sure_loss()
    # Second opinion from cddlib directly: does the constrained polytope have a vertex?
    vertices = list(cdd.Polyhedron(cut.get_matrix()).get_generators())
    assert answer == bool(vertices), (a, answer, vertices)
    return answer


def analyse(problem):
    payoff = [[Fraction(x) for x in row] for row in problem["payoff"]]
    points = problem["credal"]
    states = len(payoff[0])
    lowprev = credal_lower_prevision(points, states)
    assert lowprev.number_type == "fraction"
    assert lowprev.is_avoiding_sure_loss()
    extreme = sorted(set(tuple(Fraction(x) for x in v) for v in lowprev.get_credal_set()))
    given = set(tuple(Fraction(x) for x in point) for point in points)
    assert extreme and set(extreme) <= given, (problem["name"], extreme, given)
    gambles = [lowprev.make_gamble(row) for row in payoff]
    lower = [lowprev.get_lower(g) for g in gambles]
    upper = [lowprev.get_upper(g) for g in gambles]
    assert all(isinstance(x, (int, long, Fraction)) for x in lower + upper)  # exact
    actions = range(len(payoff))
    maximal = chosen(OptLowPrevMax(lowprev), gambles)
    interval = chosen(OptLowPrevMaxInterval(lowprev), gambles)
    return {
        "name": problem["name"],
        "payoff": problem["payoff"],
        "credal": points,
        "improb_extreme_points": [[text(x) for x in v] for v in extreme],
        "lower_expectation": [text(x) for x in lower],
        "upper_expectation": [text(x) for x in upper],
        "gamma_maximin": chosen(OptLowPrevMaxMin(lowprev), gambles),
        "maximal": [a in maximal for a in actions],
        "interval_dominance": [a in interval for a in actions],
        "e_admissible": [e_admissible(points, states, payoff, a) for a in actions],
    }


def main():
    with io.open(SOURCE, encoding="utf-8") as stream:
        source = json.load(stream)
    document = {
        "provenance": {
            "python": platform.python_version(),
            "improb": improb.__version__,
            "pycddlib": cdd.__version__,
            "arithmetic": "exact: fractions.Fraction, cddlib with GMP rationals",
            "script": "validation/improb_reference.py",
            "problems": "payoff and credal of test/fixtures/validation/finite_reference.json"
                        " (seed %s)" % source["provenance"]["seed"]},
        "conventions": {
            "indices": "gamma_maximin lists zero-based action indices, all ties",
            "credal": "LowPoly(credalset=points): the facets of the hull of the points",
            "maximal": "OptLowPrevMax: a is rejected when some b has lower prevision of"
                       " (b - a) > 0, strictly",
            "interval_dominance": "OptLowPrevMaxInterval: a is rejected when some b has"
                                  " lower(b) > upper(a), strictly",
            "e_admissible": "no improb operator; composed in the script from"
                            " LowPoly.set_lower and LowPoly.is_avoiding_sure_loss"},
        "problems": [analyse(problem) for problem in source["problems"]]}
    with io.open(OUT, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(document, separators=(",", ":"), sort_keys=True).decode("ascii")
                     + "\n")
    print("wrote %s: %d problems" % (os.path.relpath(OUT, ROOT), len(document["problems"])))


if __name__ == "__main__":
    main()
