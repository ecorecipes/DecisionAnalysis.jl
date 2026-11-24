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
| EVPPI by multivariate adaptive regression splines (`earth`); EVSI by moment matching; the variance-based measures `evppivar` and `evsivar`; `pop_voi`, `enbs` and `enbs_opt` | R `voi` 1.0.3 (with `earth` 5.3.6) | `voi_extra_reference.R` | `test/fixtures/voi/voi_extra_reference.json`, on the same draws; read by `test/test_voi_methods.jl` |
| CVaR and quantile | R `cvar` 0.6.1 | `cvar_reference.R` | `test/fixtures/validation/cvar_reference.json` |
| Maximin, maximax, Hurwicz, Laplace, minimax regret, expected utility and regret, EVPI, EVSI, quantile (exact) | textbook definitions written independently in Python with `fractions.Fraction` | `finite_reference.py` | `test/fixtures/validation/finite_reference.json` |
| CVaR (Rockafellar-Uryasev programme), lower and upper expectations, E-admissibility, maximality, interval dominance, the garbling test, Le Cam's deficiency, the upper EVPI envelope | `scipy.optimize.linprog` (HiGHS), from formulations written independently of `src/programs.jl` | `finite_reference.py` | the same file |
| Lower and upper expectations over a credal set, Γ-maximin, maximality, interval dominance, E-admissibility (exact) | Python `improb` 0.1.1 (`pycddlib` 1.0.4), under Python 2.7 | `improb_reference.py` | `test/fixtures/validation/improb_reference.json`, on the payoff tables and credal sets of `finite_reference.json` |

The finite checks run on the three worked examples of
`test/fixtures/worked_examples.json` and on 30 random problems (seed 20261123).

## Versions used

- R 4.6.0 (2026-04-24), `voi` 1.0.3, `mgcv` 1.9.4, `earth` 5.3.6, `cvar` 0.6.1, `jsonlite` 2.0.0.
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
Rscript validation/voi_extra_reference.R # a few seconds
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

### Published analyses

Five published value-of-information analyses are recomputed from the tables printed in
the papers and their supplements. Unlike the comparisons above, no script writes these
fixtures: `test/fixtures/replications/<name>.json` were transcribed from the sources
named in each file (DOI, table or figure, licence, date and place of retrieval), and
`test/test_replications.jl` asserts our result against the published number to the
precision printed. Downloaded papers and supplements are kept outside the repository
and are not copied into it; only the small input tables are, with attribution. The
unit of each is the paper's own.

| Paper | Quantity | Published | Ours | Agreement |
|:--|:--|--:|--:|:--|
| Shea et al. 2014, PLOS Biol. (CC0) | EVPI, GBP million, case 1; case 2 | 45.9; 61.2 | 45.9; 61.2 | yes, after correcting one entry of Table 2 (below) |
| | best fixed cost, case 1; case 2 | 696.0; 540.7 | 696.0; 540.6 | to rounding |
| | adaptive cost by first stage (Table S2), 8 values | | | within 0.5 (Table S1 is in whole millions) |
| | adaptive gain, case 1; case 2 | 0.78; 20.1 (32.85% of EVPI) | 0.78; 20.07 (32.8%) | yes |
| | weight on the UK kernel below which more than 80% is recovered; above which less than 2% | 0.13; 0.41 | 0.13; 0.43 | second from rounded Table S1 |
| Runge et al. 2011, Biol. Conserv. | best expected score; with perfect information; EVPI | 0.590; 0.707; 0.117 (19.9%) | 0.590; 0.707; 0.117 (19.9%) | yes |
| | share of EVPI by hypothesis tested | 54, 32, 8, 6% | 53.6, 32.2, 8.5, 5.7% | yes; they sum to the EVPI exactly |
| | fledging rate: best; with perfect information | 0.185; 0.232 (25.7%) | 0.185; 0.233 (25.7%) | to rounding |
| | EVSI / EVPI of trying a strategy, read from Figure 3 at four points | 0.31, 0.17, 0.36, 0.17 | 0.308, 0.164, 0.355, 0.170 | to reading precision |
| | error up to which killing flies is most informative and gives over 60% of EVPI | "0.03 or less" | most informative to 0.028, over 60% to 0.023 | approximately |
| Eze et al. 2023, BMC Health Serv. Res. (CC BY) | EVPI in ICU beds at 11 weights, three beliefs (authors' notebook) | 33 values | | within 0.06 (Table 4 is in whole beds) |
| | EVPI, medium belief | 11 values | | only with the notebook's prior, which sums to 0.99 |
| | EVPXI of each scenario (authors' notebook) | 27 values | | within 0.06 |
| | EVPXI of each scenario (article, Table 5) | 27 values | | **no** (below) |
| | EVPI quoted in the text: 65; 33; 91 | 65; 33; 91 | 65.1; 44.9; 81.8 | first only |
| | "per-parameter EVPI" (authors' notebook) | 33 values | | exact from the notebook's own collapsed inputs |
| Li et al. 2021, J. Appl. Ecol. (CC BY) | Table 1: EVPI; improvement | 0.15; 14.9% | 0.147; 14.8% (15.0% from unrounded growth rates) | to rounding |
| | growth rates of Table S8, 65 records by 3 models | 195 values | | within 0.0006, from matrices rebuilt from the supplement |
| | joint EVPI; biological; operational, % | 15.6; 8.5; 10.5 | 8.0; 0.0; 8.0 | **no** (below) |
| Luhede et al. 2024, Ecol. Econ. (CC BY) | value under the prior; with perfect information; EVPI, EUR million per year | 110; 119; 9 | 110; 119; 9 | exact |
| | EVPI at cost 100, prior 0.5; prior of the peak at cost 50 | 50; 0.75 | 50; 0.75 | exact |
| | value of imperfect monitoring | 2.21 | not computable | likelihood unpublished |

The authors' code was available for one of the five (Eze et al.: a Jupyter notebook
and a workbook on GitHub, without a licence). It was read, not executed, and its
printed outputs are the targets marked "authors' notebook"; nothing else is taken from
it. No R code accompanies any of the five.

What the comparisons found:

- **Shea et al. 2014, Table 2.** The entry for contiguous-premises culling under the
  fat kernel is printed as 221.1, a repeat of the entry beside it. The table's own
  weighted averages (696.0 and 540.7) need 378.1 and Table S1 has 378. With the printed
  entry the EVPI would be 6.7 and 8.9 million pounds, not the 45.9 and 61.2 of the text.
- **Eze et al. 2023.** Equation 6 as printed (minimum inside the sum) would make every
  EVPXI equal to the EVPI; the notebook computes the EVPPI of the partition {scenario},
  {the rest}, which is what we reproduce. Table 5 of the article does not follow from
  the belief in its caption (a product belief centred on *medium* values comes within
  0.63 beds in every row). The medium-belief curve comes from a prior summing to 0.99
  (a spreadsheet cell multiplies the wrong marginals). The case-severity curve of the
  "per-parameter EVPI" was computed under the belief (0.33, 0.33, 0.34), not the severe
  belief, through an index one place off. The collapsed shortfalls in the notebook are
  not the means of the article's Table 4. The text's 33 and 91 beds are 44.9 and 81.8
  in the article's own table and notebook. The statement that child infectiousness has
  no value is true of the collapsed calculation only: its EVPPI is 7.9 beds, and 16
  beds once the other two parameters are known.
- **Li et al. 2021.** With fecundity-only effects at the quartiles of Table S8, the
  best agent in each cell of the paper's Table 2 is reproduced for Australia and the
  USA and not for New Zealand (18 of 27 cells differ). Under fecundity-only effects the
  best agent is the same under every model, so the value of resolving biological
  uncertainty is necessarily zero; the published 8.5% implies that the effect of
  *T. horridus* entered the authors' calculation in another way. Unresolved: the Dryad
  deposit (doi:10.5061/dryad.83bk3j9rf) could not be retrieved.
- **No bug was found in the package.** Three functions were added because the
  replications needed them: `hypothesis_partition`, `two_stage_value` and
  `gaussian_experiment`.

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
   is compared with `voi`'s regression value and the closed form. Moment matching
   (`mm`) uses its own nested sample on each side, so the two differ by Monte Carlo
   error: on the shared draws, for 10 and 100 observations of `p1` (closed forms 0.0723
   and 0.0821), ours gave 0.0696 (standard error 0.0062) and 0.0758 (0.0056), and `voi`
   0.0929 and 0.0658 with the same number of nested data sets (30).
7. **Multivariate adaptive regression splines.** Ours places knots on a grid of
   weighted quantiles and `earth` at any draw, and the two were written independently,
   yet on the shared draws they agree to the third decimal: EVPPI of `p1` 0.06427
   against 0.06448, of `p2` 0.35674 against 0.35677, of both 0.44106 against 0.44106.
8. **Estimation.** The variance-based measures agree with `voi`'s: expected reduction
   in the variance of `p1 − p2` from learning `p1`, 0.997 against 0.989 (closed form
   1), from learning `p2`, 3.958 against 3.958 (closed form 4), and from 10 and 100
   observations of `p1`, 0.873 and 0.975 against 0.895 and 0.984 (closed forms 0.909
   and 0.990; each side simulates its own data).
9. **Net benefit of sampling.** `enbs_curve` reproduces `voi::enbs` to rounding (ENBS,
   its standard deviation from the costs, the probability that the study is worth its
   cost) and `optimal_sample_size` its optimum and near-optimal range. `voi` reads a
   95% interval for a cost as four standard deviations wide, and so do we. One
   difference is deliberate: at a zero discount rate `pop_voi` returns value ×
   population, and `population_value` returns value × population × horizon, the limit
   of the discounted formula as the rate goes to zero. The smoothed optimum uses a
   regression spline here and `mgcv` there, and agrees to within the flat top of the
   curve.
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
- `voi`'s chemotherapy EVSI examples and its `bart` and `inla` regressions, which are
  not implemented here (INLA is an R package; the built-in Gaussian process serves the
  same case).

### Published analyses not replicated

Of the 14 papers of `lab/reviews/voi_epidemiology_ecology_review.md`, nine were not
replicated. Impalli et al. 2025 (PLOS Comput. Biol.) and the measles case study of
Shea et al. 2014 print no table of outcomes: their targets exist only as figures, and
each needs the authors' simulation model. Meeyai et al. 2015 and Phillips et al. 2023
publish model code but not the probabilistic-sensitivity output that a
value-of-information calculation starts from; Kim et al. 2017 and Zang et al. 2020 are
author manuscripts without either. Christensen et al. 2026 and Bal et al. 2018 are
free to read at the publisher, which refuses automated retrieval, and have no copy in
an open repository; Moore and Runge 2012 and Raymond et al. 2020 are behind a paywall.

