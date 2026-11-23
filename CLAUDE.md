# DecisionAnalysis.jl

Standard decision analysis on finite problems: payoff tables, beliefs, rules, Savage regret, experiments
and the exact value of information. The Catlab-free core of the decision layer of the ecorecipes ecosystem.

## Place in the ecosystem

Model packages produce outcomes; this package analyses the payoff table they give. It depends on
FiniteKernels (finite spaces, kernels and states) through a `[sources]` path to the sibling directory
`../FiniteKernels.jl`, and on `LinearAlgebra`, `Random`, `Statistics` and `Tables`. FiniteKernels brings
ResilienceCore and StructuredPopulationCore by URL; Pkg follows the sources of a path dependency, so they
are not listed here. The planned `CategoricalDecisionAnalysis.jl` sits one level up, imports Catlab
generics, forwards to this package and re-exports its names (ADR 0009). No model package depends on this
one; adapters will be weak-dependency extensions on this side.

## Invariants that must not be broken

- **No Catlab, GATlab or ACSets, ever**, in `[deps]`, `[extras]` or the docs environment.
- Never define or import `compose`, `otimes`, `id`, `mcopy`, `delete`, `braid`, `dom` or `codom`, and never
  export a bare `regret` or `regrets` (`CategoricalInfluenceDiagrams` exports `regrets` for a different
  quantity). `test/test_docstrings.jl` checks all of these.
- **Layouts.** `PayoffTable.Q[a, θ]`: rows = actions, columns = states. `Experiment.likelihood[z, θ] =
  P(z | θ)`: rows = signals, columns = states, columns sum to one. A garbling `G[z′, z]` is
  column-stochastic and acts as `G * L`.
- **The value-of-information definitions are a contract with the Lean development** and are written at the
  top of `src/voi.jl`: `value(e) = Σ_z max_a Σ_θ p(θ) L[z, θ] Q[a, θ]` (joint form, no division, so a signal
  of probability zero contributes zero), `V0 = max_a Σ_θ p(θ) Q[a, θ]`, `EVSI = value(e) − V0`,
  `EVPI = Σ_θ p(θ) max_a Q[a, θ] − V0`, `EVPPI = Σ_cells max_a Σ_{θ ∈ cell} p(θ) Q[a, θ] − V0`, and regret
  `R[a, θ] = max_a′ Q[a′, θ] − Q[a, θ]`. Change neither side alone.
- **The schemas `decision-certificate/1`, `garbling-witness/1` and `e-admissible-witness/1` and the fixture
  file `test/fixtures/worked_examples.json` are the same kind of contract.** The schemas are written down in
  `docs/src/certificates.md` and checked by `proofs/` (`lake exe check_certificate`). A witness is verified
  in exact arithmetic before it is written. A change of meaning needs a new schema string.
- Exact in, exact out. No function converts a `Rational` problem to floating point; tolerances default to
  zero for exact element types. The built-in linear-programme solver works in `Rational{BigInt}` for every
  input.
- Robust value of information is returned as computed, never clamped at zero. The rule-specific value and
  the EVPI envelope are separate fields of `RobustVOI`.
- E-admissibility is decided by linear programme over the credal set. Enumerating extreme points is wrong
  and a test (`test/test_credal.jl`) shows it.
- An operation that is not defined for a belief form throws `BeliefFormError`; it never guesses a prior.
  `Hurwicz` is over states and `AlphaMaxmin` over priors, with opposite conventions for `α`, both
  documented.
- Ties: every tied action is returned, in table order, and `first_choice` is the first. Nothing is random.
- A baseline changes no score, choice or value; it only adds the comparison with no intervention.
- Types are immutable. Constructors copy their arrays and check them.

## Layout

- `src/errors.jl`: the root `DecisionAnalysisError` and every exception under it, with `showerror`.
- `src/lp.jl`: internal dense two-phase simplex in exact rational arithmetic (Bland's rule).
- `src/beliefs.jl`: `PointBelief`, `FiniteBelief`, `CredalSet`, `ScenarioSet`; label and tolerance helpers.
- `src/payoff.jl`: `PayoffTable`.
- `src/problem.jl`: `DecisionProblem`, feasibility filter, baseline and gains over it,
  `restrict_scenarios`, and the internal `_View` that aggregators act on.
- `src/aggregators.jl`: the ten aggregators and `aggregate`.
- `src/regret.jl`: regret tables and profiles.
- `src/rules.jl`: transforms, `RobustnessMetric`, selectors, `DecisionRule`, the named rules, the
  set-valued rules, `DecisionResult`, `decide`.
- `src/experiments.jl`: `Experiment`, `bayes_invert`, `garble`, `is_garbling_of`, `deficiency`.
- `src/voi.jl`: `evpi`, `evppi`, `evsi`, `enbs`, `value_with_information`, `robust_evpi`, `robust_evsi`.
- `src/fidelity.jl`: `total_variation`, `decision_loss_bound`.
- `src/report.jl`, `src/show.jl`: `DecisionReport`, its Tables.jl rows, and printing.
- `src/certificates.jl`: `export_decision_certificate`, `export_garbling_witness`,
  `export_eadmissible_witness` and their hand-written JSON encoder.
- `proofs/`: the Lean 4 / Mathlib project `DecisionAnalysisProofs` and its `check_certificate` executable;
  see `proofs/README.md`.
- `test/helpers.jl`: the fixture reader and the slow oracles; `test/test_*.jl`: the suite, seeded
  `Random.Xoshiro` only; `test/fixtures/worked_examples.json`: the three worked examples.
- `docs/`: Documenter manual (`index.md`, `api.md`, `certificates.md`).

## Commands

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the test suite; the only run that counts
julia --project=docs docs/make.jl --no-deploy                    # build the docs locally
DECISIONANALYSIS_LEAN=1 julia --project -e 'using Pkg; Pkg.test()'   # also run exported files through the Lean checker (needs lake and a built proofs/)
```

## Files not to edit by hand

- `Manifest.toml` is generated and ignored.
- `docs/build/` is generated.
- `test/fixtures/worked_examples.json` is shared with the Lean development. Edit it only together with
  that side, and keep every number in it equal to what `lab/wiki/references/computations/worked_examples.py`
  prints.

## Style

JuliaFormatter `yas`; `snake_case` functions, `CamelCase` types, a leading underscore for internals; British
spelling in prose and American in identifiers; no emojis. A docstring on every exported name, enforced by
`test/test_docstrings.jl`. Typed exceptions in `src/errors.jl` under `DecisionAnalysisError`, carrying the
offending names (ADR 0013); invalid arguments and keywords raise `ArgumentError`; never a bare `error()`;
typed errors from FiniteKernels pass through unchanged; another package's type is a code span, never
`@ref`. Every fast path is tested against a slow oracle in `test/helpers.jl` that shares no code with it.
