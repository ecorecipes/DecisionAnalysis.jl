# Proposed schemas for certified "no" answers

Status: **proposed**. The Lean checkers exist (`DecisionAnalysisProofs/Certificate/Refutation.lean`),
are wired into `lake exe check_certificate`, and have hand-made fixtures in
`certificate-fixtures/`. The Julia exporters do not exist yet, and `docs/src/certificates.md`
does not describe these schemas. When the exporters are written, move the two sections below
into `docs/src/certificates.md` and delete this file.

Today a "not E-admissible" answer of `decide(problem, EAdmissible())` and a "not a garbling"
answer of `garbling_result` are exact only when they come from an exact route; from a
floating-point optimiser they are flagged numerical (`details.exact`, `GarblingResult.exact`).
With the witnesses below a negative answer can be verified in exact arithmetic whatever found
it, as positive answers already are, and both are **complete**: whenever the answer is "no", a
witness of the stated form exists (Lean theorems `not_isEAdmissible_iff_isMixedDominated` and
`not_isGarblingOf_iff_exists_separating`).

The encoding is the one common to the existing schemas: every number is a JSON string holding
an exact rational (`"n/d"` in lowest terms with `d > 1`, or `"n"`); indices are JSON integers
counted from zero; matrices are arrays of rows; payoff rows are actions and columns are states;
likelihood rows are signals and columns are states. Exit codes of the checker: 0 accepted,
1 rejected, 2 cannot be decoded.

## Schema `not-e-admissible-witness/1`

Establishes that one action is **not** E-admissible over a credal set: no prior in the convex
hull of the extreme points makes it a Bayes action.

| member | type | meaning |
|---|---|---|
| `schema` | string | always `"not-e-admissible-witness/1"` |
| `payoff` | `m` arrays of `n` numbers | `Q[a][θ]`, rows = feasible actions, columns = states |
| `priors` | `K` arrays of `n` numbers | the extreme points of the credal set, one per row, each nonnegative and summing to one |
| `action` | integer in `0 … m−1` | the action claimed not to be E-admissible |
| `mixed_action` | array of `m` numbers | `μ[b]`, a probability vector over the actions |

Labels (`actions`, `states`) may be added as arrays of strings; the checker ignores members it
does not know.

**What the checker verifies** (`Certificate.checkNotEAdmissible`):

1. `payoff` is `m × n` with `m` the length of `mixed_action` and `n` the length of its first
   row; `priors` is `K × n`.
2. Every row of `priors` is nonnegative and sums to one.
3. `mixed_action` is nonnegative and sums to one, and `action < m`.
4. For **every** extreme point `k`:
   `Σ_b μ[b] · E[b, k]  >  E[action, k]`, strictly, with `E[b, k] = Σ_θ priors[k][θ] · payoff[b][θ]`.

**Soundness** (`Certificate.RawNotEAdmissible.accepts_sound`, via
`IsMixedDominated.not_isEAdmissible`): strict dominance at every extreme point is strict
dominance at every prior of the hull, so under every such prior some pure action is strictly
better than `action`.

**Completeness** (`not_isEAdmissible_iff_isMixedDominated`): every action that is not
E-admissible has such a `mixed_action`. An action rejected already by `Maximal()` has a witness
with a single 1 (`isMixedDominated_of_not_isMaximal`).

**Where Julia finds the witness.** In the programme of `eadmissibility_model`,

    maximise t  subject to  Σ_k λ_k = 1,  Σ_k λ_k (E[a, k] − E[b, k]) − t ≥ 0 for every b,  λ ≥ 0,

the dual is: minimise `s` subject to `Σ_b μ_b (E[a, k] − E[b, k]) ≤ s` for every `k`,
`Σ_b μ_b = 1`, `μ ≥ 0`. So the multipliers `μ_b` of the rows `optimal[b]` at the optimum are
already a probability vector, and the optimum `t* < 0` says
`Σ_b μ_b E[b, k] − E[a, k] ≥ −t* > 0` for every `k`. `solve_program` already rounds and checks
the dual solution on the floating-point route (`_dual_certifies`); the exporter needs the
multipliers of `constraints.optimal` (up to the sign convention of the solver), cleaned as
weights are (negative entries to zero, renormalised), and then condition 4 checked exactly
before writing. A strict margin survives rounding whenever `−t*` is larger than the rounding
error, which is the same condition under which the "no" was believable in the first place.

Example (`certificate-fixtures/not_eadmissible_mixed.json`, the "hedge" instance of
`test/test_credal.jl`): the third action is maximal (no pure action dominates it) and still not
E-admissible.

```json
{"schema": "not-e-admissible-witness/1",
 "payoff": [["1", "0"], ["0", "1"], ["2/5", "2/5"]],
 "priors": [["1", "0"], ["0", "1"]],
 "action": 2,
 "mixed_action": ["1/2", "1/2", "0"]}
```

## Schema `not-garbling-witness/1`

Establishes that the experiment `garbled` is **not** a garbling of `experiment`, by exhibiting
a decision problem in which deciding after `garbled` is worth strictly more.

| member | type | meaning |
|---|---|---|
| `schema` | string | always `"not-garbling-witness/1"` |
| `experiment` | `k` arrays of `n` numbers | `L[z][θ]`, the likelihood of `e`; every column sums to one |
| `garbled` | `k′` arrays of `n` numbers | `L′[z′][θ]`, the likelihood of `e′`, claimed not to be a garbling of `e` |
| `payoff` | `m ≥ 1` arrays of `n` numbers | the separating payoff table `Q[a][θ]` |
| `prior` | array of `n` numbers | the prior of the separating problem, nonnegative, summing to one |

The roles of `experiment` and `garbled` are those of `garbling-witness/1`: the positive
certificate says `garbled = G · experiment`; this one says no such `G` exists.

**What the checker verifies** (`Certificate.checkNotGarbling`):

1. Shapes: `n` is the length of the first row of `experiment`; `garbled` and `payoff` have `n`
   columns; `prior` has `n` entries; `payoff` has at least one row.
2. Both likelihoods are column-stochastic; the prior is nonnegative and sums to one.
3. `value(garbled) > value(experiment)`, strictly, both recomputed as
   `Σ_z max_a Σ_θ prior[θ] · L[z][θ] · payoff[a][θ]` (the joint form of `value_with_information`).

**Soundness** (`Certificate.RawNotGarbling.accepts_sound`): by T6
(`valueWithInformation_garble_le`) a garbling is never worth more, for any payoff table and
any prior. No duality is involved.

**Completeness** (`not_isGarblingOf_iff_exists_separating`, Blackwell's theorem): whenever
`e′` is not a garbling of `e`, a separating payoff table exists with **one action per signal
of `e′`** (`m = k′`) under **any prior of full support**, in particular the uniform prior. A
prior with a zero entry may fail to separate (states of probability zero are invisible), so the
exporter should use the uniform prior.

**Where Julia finds the witness.** The programme of `garbling_model` is infeasible exactly when
its Farkas alternative is feasible: there are `v[z′, θ]` (multipliers of the rows
`product[z′, θ]`) and `c[z]` (multipliers of `columns[z]`) with

    Σ_θ L[z, θ] v[z′, θ] + c[z] ≤ 0   for every z′ and z,
    Σ_{z′, θ} L′[z′, θ] v[z′, θ] + Σ_z c[z] > 0.

Take the actions to be the signals of `e′`, the prior uniform, and `payoff[z′][θ] = n · v[z′, θ]`.
Then `value(e) = Σ_z max_{z′} Σ_θ L[z, θ] v[z′, θ] ≤ −Σ_z c[z]`, and
`value(e′) ≥ Σ_{z′} Σ_θ L′[z′, θ] v[z′, θ] > −Σ_z c[z]`. The built-in simplex reaches the
verdict in phase one, whose final dual vector is this certificate; a floating-point optimiser
reports it as an infeasibility ray (`JuMP.dual` with `MOI.INFEASIBILITY_CERTIFICATE` as dual
status). Alternatively solve `deficiency_model` and read the same multipliers from its optimal
dual. Either way the rounded table needs only condition 3 checked exactly before writing; any
payoff table that passes is a valid witness, so no cleaning step can make it wrong.

Example (`certificate-fixtures/not_garbling_flood.json`): the river gauge is not a garbling of
its own garbling; the flood problem itself separates them.

```json
{"schema": "not-garbling-witness/1",
 "experiment": [["37/50", "14/25", "8/25", "23/100"], ["13/50", "11/25", "17/25", "77/100"]],
 "garbled": [["9/10", "3/5", "1/5", "1/20"], ["1/10", "2/5", "4/5", "19/20"]],
 "payoff": [["0", "-10", "-60", "-150"], ["-20", "-20", "-25", "-90"], ["-48", "-48", "-48", "-48"]],
 "prior": ["1/2", "3/10", "3/20", "1/20"]}
```

## Suggested Julia surface

* `export_not_eadmissible_witness(path, problem, action; optimizer, rounding_tol)` and
  `export_not_garbling_witness(path, e′, e; optimizer, rounding_tol)`, each verifying its
  condition exactly before writing and throwing `ExactVerificationError` otherwise.
* `details.exact[action]` of an `EAdmissible` result and `GarblingResult.exact` can then be
  `true` for a "no" from a floating-point optimiser whenever the rounded dual passes the exact
  check, instead of always `false`; `details` gains `dominating[action]` (the mixed action) and
  `GarblingResult` a `separating` problem.
* With `DECISIONANALYSIS_LEAN=1`, run the exported files through `check_certificate` as the
  existing witnesses are.
