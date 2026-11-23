# Contributing to DecisionAnalysis.jl

## Local setup

The ecosystem packages depend on each other through `[sources]` entries that point at sibling directories.
Clone this package beside FiniteKernels.jl:

```sh
git clone https://github.com/ecorecipes/FiniteKernels.jl
git clone https://github.com/ecorecipes/DecisionAnalysis.jl
cd DecisionAnalysis.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

`Pkg.test()` is the only run that counts as "the suite passes".

## Style

- Julia ≥ 1.12. Every dependency, including stdlibs and test-only extras, has a `[compat]` entry.
- Format with JuliaFormatter using the repository `.JuliaFormatter.toml` (`style = "yas"`).
- `snake_case` for functions and variables, `CamelCase` for types, a leading underscore for internal
  helpers. British spelling in prose, American in identifiers. No emojis.
- Public names have docstrings with a signature line, a one-sentence summary, the conventions that fix the
  meaning (layouts, signs, which way `α` points), and an example where practical.
- Errors follow ADR 0013. Exceptions are typed, carry the offending labels, and live in `src/errors.jl`
  under `DecisionAnalysisError`; library code never calls a bare `error("...")`. Invalid arguments and
  keywords raise `ArgumentError`. A typed error from FiniteKernels passes through unchanged.
- Types are immutable; constructors copy and check their arrays.
- No Catlab, GATlab or ACSets. No `compose`, `otimes`, `id`, `mcopy`, `delete`, `braid`, `dom`, `codom`,
  `regret` or `regrets`.

## Tests

- One `@testset` in `test/runtests.jl` including `test_*.jl`. Random tests use a seeded `Random.Xoshiro`.
- Every fast path is checked against a slow oracle written from the definition, in `test/helpers.jl` or
  beside the test, sharing no code with the package.
- Exact arithmetic first: a property is asserted with `==` on `Rational{BigInt}` problems, then
  approximately in `Float64`.
- A new rule or aggregator needs a worked value by hand, a random-problem test against its definition, and
  a test of the belief forms it refuses.
- Do not weaken a failing test. Fix the code, or record the failure.

## Contracts with the Lean development

The value-of-information definitions (top of `src/voi.jl`), the certificate schema
(`docs/src/certificates.md`) and the fixture file (`test/fixtures/worked_examples.json`) are shared with
the proofs. Change them only together with that side; a change of meaning in a certificate needs a new
schema string.

## Pull requests

Branch from `main`, keep pull requests focused, and make sure CI is green (tests on Julia 1.12 and latest,
formatter check, docs build).
