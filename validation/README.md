# Validation against other software

Comparisons of DecisionAnalysis.jl with established R packages, with independent
Python code and with the Python package `improb`. The scripts here write small fixture
files, with their provenance, under `test/fixtures/`; the Julia test suite
(`test/test_validation.jl`, `test/test_validation_improb.jl`) only reads those
fixtures, so it needs neither R nor Python.

Money appears in this directory and in the fixtures only as a published validation
target (the chemotherapy example of the R package `voi`, in pounds). It is used nowhere
else in the package: utilities are counted in individuals.

## What is compared

| Quantity | Reference | Script | Fixture |
|:--|:--|:--|:--|
| EVPI; EVPPI by Strong and Oakley, Sadatsafavi, generalised additive model, Gaussian process; EVSI by regression and importance sampling | R `voi` 1.0.3 (with `mgcv` 1.9.4) | `voi_reference.R` | `test/fixtures/voi/voi_reference.json`, on the draws of `test/fixtures/voi/normal_sample.csv` (written by `make_sample.jl`) |
| CVaR and quantile | R `cvar` 0.6.1 | `cvar_reference.R` | `test/fixtures/validation/cvar_reference.json` |
| Maximin, maximax, Hurwicz, Laplace, minimax regret, expected utility and regret, EVPI, EVSI, quantile (exact) | textbook definitions written independently in Python with `fractions.Fraction` | `finite_reference.py` | `test/fixtures/validation/finite_reference.json` |
| CVaR (Rockafellar-Uryasev programme), lower and upper expectations, E-admissibility, maximality, interval dominance, the garbling test, Le Cam's deficiency, the upper EVPI envelope | `scipy.optimize.linprog` (HiGHS), from formulations written independently of `src/programs.jl` | `finite_reference.py` | the same file |
| Lower and upper expectations over a credal set, Γ-maximin, maximality, interval dominance, E-admissibility (exact) | Python `improb` 0.1.1 (`pycddlib` 1.0.4), under Python 2.7 | `improb_reference.py` | `test/fixtures/validation/improb_reference.json`, on the payoff tables and credal sets of `finite_reference.json` |

The finite checks run on the three worked examples of
`test/fixtures/worked_examples.json` and on 30 random problems (seed 20261123).

## Versions used

- R 4.6.0 (2026-04-24), `voi` 1.0.3, `mgcv` 1.9.4, `cvar` 0.6.1, `jsonlite` 2.0.0.
- Python 3.9.6, `numpy` 2.0.2, `scipy` 1.13.1 (`requirements.txt`).
- Python 2.7.15 (conda-forge, `osx-64`, under Rosetta on Apple silicon), `improb` 0.1.1,
  `pycddlib` 1.0.4 (its bundled cddlib, built against conda-forge `gmp` 6.3.0 with
  Apple clang 21.0.0), `micromamba` 2.10.0 (`requirements-py27.txt`).
- Julia 1.12.6, GAM.jl 0.3.0 (for `GAMRegression`).

## How to run

From the package root:

```sh
# R: packages go to the user library
Rscript -e 'install.packages(c("voi", "cvar", "jsonlite"), repos = "https://cloud.r-project.org")'
julia validation/make_sample.jl          # only to regenerate the shared draws
Rscript validation/voi_reference.R       # about half a minute
Rscript validation/cvar_reference.R

# Python: a local virtual environment (ignored by git)
python3 -m venv validation/.venv
validation/.venv/bin/pip install -r validation/requirements.txt
validation/.venv/bin/python validation/finite_reference.py

# Python 2.7 with improb: a second local environment (ignored by git); see below
validation/.venv27/bin/python validation/improb_reference.py

# Julia: the suite reads the fixtures
julia --project -e 'using Pkg; Pkg.test()'
# ... and, with R and voi installed, also compares on voi's chemotherapy data
DECISIONANALYSIS_R=1 julia --project -e 'using Pkg; Pkg.test()'

# The side-by-side table below (needs an environment with DecisionAnalysis, GAM and JSON)
mkdir -p /tmp/chemo && Rscript validation/chemo_data.R /tmp/chemo
julia --project=<environment> validation/compare_voi.jl /tmp/chemo
```

The chemotherapy data (`chemo_nb`, `chemo_pars`) ship with `voi`, which is licensed
under the GPL-3, and are not copied into this repository: `chemo_data.R` writes them to
a temporary directory from the installed package.

### The Python 2.7 environment for `improb`

`improb` 0.1.1 (2011) is Python 2 code and needs `pycddlib` of its time. What worked,
on macOS 26 on Apple silicon, without sudo and without touching the system Python
(there is no Python 2.7 for `osx-arm64` on conda-forge, hence `osx-64` and Rosetta; on
an Intel machine or on Linux leave out `--platform` and `-arch x86_64`):

```sh
# A self-contained micromamba binary, anywhere outside the repository
mkdir -p "$TMPDIR/mm" && curl -sL https://micro.mamba.pm/api/micromamba/osx-arm64/latest |
    tar -xj -C "$TMPDIR/mm" bin/micromamba
export MAMBA_ROOT_PREFIX="$TMPDIR/mamba-root"      # package cache; not in $HOME
"$TMPDIR/mm/bin/micromamba" create -y --no-rc -p validation/.venv27 -c conda-forge \
    --platform osx-64 python=2.7 pip gmp

# pycddlib 1.0.4 from its source archive (it bundles cddlib), against that GMP
P="$PWD/validation/.venv27"
CC="clang -arch x86_64" ARCHFLAGS="-arch x86_64" \
CFLAGS="-I$P/include -Wno-error=implicit-function-declaration -Wno-error=incompatible-function-pointer-types -Wno-error=int-conversion" \
LDFLAGS="-L$P/lib -Wl,-rpath,$P/lib" \
    "$P/bin/python" -m pip install --no-binary :all: pycddlib==1.0.4
# improb imports cdd in its setup.py, so it comes second, without dependency resolution
"$P/bin/python" -m pip install --no-deps --no-binary :all: improb==0.1.1
```

Both builds succeeded at the first attempt with these pins (pip 20.1.1, the last line
of pip for Python 2.7); no other `pycddlib` version was tried, and Cython is not needed
because the source archive of `pycddlib` 1.0.4 ships its generated C. The conda-forge
package `cddlib` is not needed. micromamba leaves two small registry files in the home
directory (`~/.conda/environments.txt`, `~/.mamba/pkgs/urls.txt`) and pip its download
cache.

## Results

### Credal rules against `improb`

33 problems (the three worked examples, of which `flood` has a credal set and the other
two a single prior, and the 30 random problems), 115 actions: 230 lower and upper
expectations, 345 yes or no answers (maximal, interval-dominant, E-admissible) and 33
Γ-maximin choices, **no disagreement, and no change to `src/`**. Both sides are exact,
so every comparison is an equality of rationals or of sets, with no tolerance. The
problems tell the rules apart: in 3 the E-admissible set is smaller than the maximal
set, in 8 interval dominance keeps more than maximality, one has a Γ-maximin tie and
one a redundant generating point.

How the same credal set is given to both sides. A `CredalSet` is a list of generating
points. `improb` holds a lower prevision as bounds on expectations; the set is built
as `LowPoly(pspace, credalset=points, number_type='fraction')`, which hands the points
to cddlib, takes the facets of their convex hull and stores one bound per facet. Lower
and upper expectations are then `improb`'s natural extension, a linear programme over
the facets (`LowPoly.get_lower`, `get_upper`), where ours are a minimum and a maximum
over the points: the two routes share nothing. The script turns the facets back into
extreme points and the fixture records them; the test checks that they are among our
points and that our results on them alone are unchanged.

Conventions, the same on both sides:

- Γ-maximin (`OptLowPrevMaxMin`): every action that attains the largest lower
  expectation is returned.
- Maximality (`OptLowPrevMax`): `a` is rejected when some `b` has a **strictly**
  positive lower expectation of `b − a`; ours rejects `a` when some `b` is strictly
  better at every generating point, which is the same statement. Two actions with the
  same expectations everywhere are both kept.
- Interval dominance (`OptLowPrevMaxInterval`): `a` is rejected when some `b` has lower
  expectation **strictly** above the upper expectation of `a`; ours keeps `a` when its
  upper expectation is at least the largest lower expectation.
- **E-admissibility is not an operator of `improb`** (neither in 0.1.1 nor on the
  master branch of `mcmtroffaes/improb`; its `OptAdmissible` is pointwise dominance
  between gambles, another notion). The script composes it from `improb`'s primitives:
  `a` is E-admissible exactly when the lower prevision of the credal set with the
  added assessments "the lower expectation of `a − b` is at least 0" for every other
  `b` avoids sure loss (`LowPoly.set_lower`, `LowPoly.is_avoiding_sure_loss`), that is,
  when the set of priors under which `a` is optimal is not empty. The arithmetic is
  `improb`'s and cddlib's; the reduction is the script's, so this row is a weaker
  check than the other four.

### Finite rules and linear programmes

1483 comparisons on 33 problems, no disagreement. The exact quantities are equal as
rationals; the quantities solved by `linprog` agree to `1e-7`, and every yes or no
(E-admissible, maximal, interval-dominant, garbling) is the same. CVaR and the lower
quantile agree with `cvar` on the worked examples (28 rows; expected shortfall to
`1e-4`, since `cvar` integrates the quantile function numerically; value at risk
exactly).

### Sample-based estimators against `voi`

Normal model of the `voi` vignette (payoffs 0 and p1 − p2), the same 2000 draws on both
sides. "s.e." is the standard error reported by this package.

| Quantity | Ours | `voi` | Our s.e. | Difference |
|:--|--:|--:|--:|--:|
| EVPI | 0.44106433 | 0.44106433 | 0.0201 | 3e-13 |
| EVPPI p1, Strong and Oakley, 20 blocks | 0.081291167 | 0.081291167 | 0.0142 | 1e-14 |
| EVPPI p2, Strong and Oakley, 20 blocks | 0.35115573 | 0.35115573 | 0.0214 | 4e-13 |
| EVPPI p1, Sadatsafavi, 1 separator | 0.084406414 | 0.084406414 | 0.0146 | 2e-15 |
| EVPPI p2, Sadatsafavi, 1 separator | 0.35821184 | 0.35821184 | 0.0210 | 5e-13 |
| EVPPI p1, Sadatsafavi, 2 separators | 0.084406414 | 0.084406414 | 0.0146 | 2e-15 |
| EVPPI p2, Sadatsafavi, 2 separators | 0.35821184 | 0.35821184 | 0.0210 | 5e-13 |
| EVPPI p1, GAM (GAM.jl; `mgcv`) | 0.069319861 | 0.069330048 | 0.0157 | −1.0e-5 |
| EVPPI p2, GAM | 0.35982196 | 0.35994357 | 0.0207 | −1.2e-4 |
| EVPPI p1, built-in spline against `voi`'s GAM | 0.075750719 | 0.069330048 | 0.0155 | 6.4e-3 |
| EVPPI p1, Gaussian process | 0.069330044 | 0.069938964 | 0.0157 | −6.1e-4 |
| EVPPI p2, Gaussian process | 0.35828473 | 0.35994195 | 0.0208 | −1.7e-3 |
| EVPPI p1 and p2, GAM tensor | 0.44106433 | 0.44106433 | 0.0201 | 3e-13 |
| EVPPI p1 and p2, Gaussian process | 0.44106433 | 0.44106433 | 0.0201 | 7e-11 |
| EVSI of 10 observations of p1, regression (closed form 0.0723251) | 0.0651 | 0.0584 | 0.0151 | 0.0067 |
| EVSI of 10 observations, importance sampling | 0.0635 | 0.0488 | 0.0132 | 0.0147 |
| EVSI of 10 observations, Gaussian approximation (against `voi`'s regression) | 0.0659 | 0.0584 | 0.0144 | 0.0075 |
| EVSI of 100 observations, regression (closed form 0.0821176) | 0.0756 | 0.0672 | 0.0155 | 0.0084 |
| EVSI of 100 observations, importance sampling | 0.0733 | 0.0566 | 0.0138 | 0.0167 |
| EVSI of 100 observations, Gaussian approximation | 0.0747 | 0.0672 | 0.0154 | 0.0075 |

Six parameters, payoffs 0 and p1 − p2 + (x3 + x4 + x5 + x6) / 2, the same draws. `voi`
chooses its Gaussian process for five focal parameters, as `evppi` does here.

| Quantity | Ours | `voi` | Our s.e. | Difference |
|:--|--:|--:|--:|--:|
| EVPI | 0.51015865 | 0.51015865 | 0.0228 | 9e-14 |
| EVPPI p1, x3 to x6, default (Gaussian process, 1500 points) | 0.17197393 | 0.17022975 | 0.0221 | 1.7e-3 |
| EVPPI p1, x3 to x6, Gaussian process on all 2000 points | 0.17254114 | 0.17022975 | 0.0224 | 2.3e-3 |
| EVPPI p2, x3 to x6, Gaussian process | 0.43864524 | 0.43706357 | 0.0235 | 1.6e-3 |
| EVPPI all six, Gaussian process | 0.51015865 | 0.51015865 | 0.0228 | 3e-11 |

The chemotherapy example bundled with `voi` (10,000 draws, 14 parameters; net benefit in
pounds at 20,000 pounds per QALY; a validation target only).

| Quantity | Ours | `voi` | Our s.e. | Difference |
|:--|--:|--:|--:|--:|
| EVPI (368.6051 in `voi`'s vignette) | 368.6051 | 368.6051 | 5.38 | −1e-9 |
| EVPPI `p_side_effects_t1`, Strong and Oakley, 50 blocks | 38.497408 | 38.497408 | 6.12 | −5e-9 |
| EVPPI `p_side_effects_t1`, Sadatsafavi, 1 separator | 32.352557 | 32.352557 | 4.92 | 5e-8 |
| EVPPI `p_side_effects_t1`, Sadatsafavi, 2 separators | 32.520482 | 32.520482 | 4.93 | 4e-8 |
| EVPPI `p_side_effects_t1`, GAM | 26.15845 | 26.754625 | 5.68 | −0.60 |
| EVPPI `logor_side_effects`, Strong and Oakley, 50 blocks | 333.30938 | 333.30938 | 5.58 | −3e-9 |
| EVPPI `logor_side_effects`, Sadatsafavi, 1 separator | 333.50404 | 333.50404 | 5.58 | 5e-8 |
| EVPPI `logor_side_effects`, Sadatsafavi, 2 separators | 333.50404 | 333.50404 | 5.58 | 2e-8 |
| EVPPI `logor_side_effects`, GAM | 329.71456 | 331.86758 | 5.57 | −2.15 |
| EVPPI `logor_side_effects`, Gaussian process, first 1500 draws | 328.47937 | 329.81334 | 14.3 | −1.33 |
| EVPPI `u_hospital`, Sadatsafavi, 1 separator | 6.9508085 | 6.9508085 | 4.44 | 4e-8 |
| EVPPI `u_hospital`, GAM | 4.2297831 | 4.6710089 | 5.16 | −0.44 |
| EVPPI of the pair, GAM tensor | 332.34506 | 333.87573 | 5.58 | −1.53 |
| EVPPI of five parameters, Gaussian process, first 1500 draws | 339.22751 | 339.35972 | 14.1 | −0.13 |

## Findings

1. **A defect on our side, fixed.** `Sadatsafavi(; separators=2)` searched the two cuts
   over a grid of at most 200 positions, and restricted the one-cut candidates to the
   same grid. On a large sample its value could therefore be **below** the
   one-separator value (0.08304 against 0.08441 for p1 above, and 32.09 against 32.35
   for `p_side_effects_t1`), which a two-cut search must never be, and it differed from
   `voi`. The search is now exhaustive for one and for two cuts, in time proportional to
   draws × actions (a running maximum per action for the middle segment), and the
   `grid` keyword is gone. A brute-force test over every pair of cuts holds it, and the
   values above now agree with `voi` to rounding.
2. **Agreement to rounding** for EVPI, `StrongOakley` (`so`) and `Sadatsafavi` (`sal`)
   with one separator, before any change. `voi`'s `sal` places cuts pair of actions by
   pair of actions; with two actions, as in both examples, that is the exhaustive
   search. With more than two actions the two can differ, and ours is then the larger.
3. **Regressions agree closely, not exactly**, as different software fits the smooths:
   GAM.jl against `mgcv` within 0.02% on the normal model and within 0.4 of a standard
   error on the chemotherapy example; our Gaussian process against `voi`'s within 1.3%
   and within 0.1 of a standard error. No change was needed.
4. **EVSI** estimates differ by Monte Carlo error only: each side simulates its own
   data. All are within about one standard error of each other and of the closed forms.
   `voi` has no Gaussian approximation in the sense of Jalal and Alarid-Escudero; ours
   is compared with `voi`'s regression value and the closed form. `voi`'s moment
   matching (`mm`) is not implemented here and was not compared.
5. **A weakness of our Gaussian process, found while writing a vignette and fixed.** On
   a five-parameter example the search for the hyperparameters settled on length scales
   shorter than the spacing of the points. Such a process is independent noise at the
   points it is fitted to, so the fitted values reproduced the payoffs and the estimate
   came out at the EVPI of the sample (0.564 where the closed form is 0.480). The length
   scales are now bounded below by the median distance between neighbouring points
   (`min_length_scale`), the estimate on that example is 0.496, and a test holds it. The
   comparison with `voi` above was made after this change.
6. **`voi`'s Gaussian process** conditions on the first `maxSample = 5000` draws by
   default. On the 10,000 chemotherapy draws that needed more than 18 Gb of memory here
   and stopped, and with `maxSample` below the number of draws `voi` 1.0.3 fails
   ("number of items to replace is not a multiple of replacement length"). Both look
   like limits of `voi`, not of this package; the Gaussian-process rows therefore use
   the first 1500 draws on both sides. Ours takes a seeded random subsample
   (`max_points`, default 1500) and predicts at every draw.
7. **Conventions that differ**, now stated in the docstrings: `Hurwicz(α)` puts `α` on
   the best outcome (some texts put it on the worst); `CVaR(α)` and `Quantile(α)` are
   for utilities, the negatives of the expected shortfall and value at risk of
   loss-based software such as `cvar`.

## What could not be compared

- No CRAN package was found for the payoff-table criteria (maximin, maximax, Hurwicz,
  Laplace, minimax regret) or for the imprecise-probability rules (Γ-maximin,
  maximality, E-admissibility): a search of the titles and descriptions of all CRAN
  packages for those terms found only design-of-experiments packages. The payoff-table
  criteria are therefore checked against independent Python code only. The
  imprecise-probability rules are checked against the Python package `improb`, which
  does not build on Python 3 (version 0.1.1 is from the Python 2 era) and runs in a
  Python 2.7 environment of its own (above); it has no operator for E-admissibility,
  which is composed from its primitives.
- Robust value of information under a rule (`robust_evpi`'s rule-specific value,
  `robust_evsi`), the risk-attitude and multi-attribute utilities, `voi_profile` and
  `optimal_design` have no counterpart in the packages above; they are tested against
  explicit oracles in the test suite only.
- `voi`'s chemotherapy EVSI examples and its `earth`, `bart` and `inla` regressions.
