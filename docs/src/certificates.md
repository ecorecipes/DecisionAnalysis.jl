# Certificates

A certificate is a JSON document in exact rational arithmetic that an independent
checker can confirm without trusting this package. There are five kinds, each with
its own schema string:

| schema | written by | what it establishes |
|---|---|---|
| `decision-certificate/1` | [`export_decision_certificate`](@ref) | the numbers claimed for one decision problem |
| `garbling-witness/1` | [`export_garbling_witness`](@ref) | one experiment is a garbling of another |
| `e-admissible-witness/1` | [`export_eadmissible_witness`](@ref) | an action is E-admissible over a credal set |
| `not-e-admissible-witness/1` | [`export_not_eadmissible_witness`](@ref) | an action is **not** E-admissible over a credal set |
| `not-garbling-witness/1` | [`export_not_garbling_witness`](@ref) | one experiment is **not** a garbling of another |

The checker is the Lean executable in `proofs/`
(`cd proofs && lake exe check_certificate FILE.json`), which dispatches on `schema`
and exits with 0 when it accepts, 1 when it rejects and 2 when the file cannot be
decoded. This page is the definition of the formats; they are a contract, and a
change of meaning needs a new schema string.

The witnesses come from linear programmes, stated as JuMP models by
[`garbling_model`](@ref), [`eadmissibility_model`](@ref),
[`mixed_dominance_model`](@ref) and [`separating_payoff_model`](@ref). By default the package
solves them exactly for rational problems with its built-in simplex; the exporters
also take an `optimizer` keyword (see [`solve_program`](@ref)), so the programme can
be solved by an exact optimiser such as `CDDLib.Optimizer{Rational{BigInt}}` or by a
floating-point one such as `HiGHS.Optimizer`, whose solution is then rounded to
rationals. The solver is never trusted: whatever route found it, each witness is
checked in exact arithmetic before a byte is written, so an exported witness is
never an unverified floating-point solution, and the file is the same kind of object
for the Lean checker whichever route produced it. If a rounded solution does not
pass the check, [`ExactVerificationError`](@ref) is thrown and nothing is written.

## Encoding common to all three

**Numbers.** Every number is a JSON **string** holding an exact rational: `"n/d"`
with `d > 1` and the fraction in lowest terms, or `"n"` when the denominator is one.
`n` is a decimal integer with an optional leading `-`; `d` is a positive decimal
integer. There are no decimal points, exponents or spaces. Examples: `"-39/2"`,
`"18"`, `"0"`.

**Indices.** Action indices are JSON **integers** counted from zero. A set of indices
is an array in increasing order.

**Matrices** are arrays of rows. Payoff rows are actions and columns are states;
likelihood rows are signals and columns are states.

**Exactness.** Every exporter requires `Rational` or integer element types and throws
[`InexactCertificateError`](@ref) otherwise.

## Schema `decision-certificate/1`

A certificate is a JSON object with these members, in this order.

| member | type | meaning |
|---|---|---|
| `schema` | string | always `"decision-certificate/1"` |
| `actions` | array of strings, length `m` | action labels |
| `states` | array of strings, length `n` | state labels |
| `payoff` | `m` arrays of `n` numbers | `Q[a][θ]`, rows = actions, columns = states; higher is better |
| `prior` | array of `n` numbers | `p[θ]`, nonnegative, summing to one |
| `signals` | array of strings, length `k` | signal labels; present only with an experiment |
| `likelihood` | `k` arrays of `n` numbers | `L[z][θ] = P(z \| θ)`, rows = signals, columns = states; every column sums to one; present only with an experiment |
| `claims` | object | the claimed results, below |

Action indices refer to positions in `actions`.

**Actions.** Only the feasible actions of the problem are written. The baseline of a
problem, if it has one, is not part of a certificate.

## Claims

With `Q`, `p` and `L` as above, sums over `θ` running over all states, and maxima
over `a` running over all actions in the certificate:

| claim | type | definition |
|---|---|---|
| `expected_utility` | array of `m` numbers | `EU[a] = Σ_θ p[θ] · Q[a][θ]` |
| `act_now_value` | number | `V0 = max_a EU[a]` |
| `bayes_actions` | array of indices | all `a` with `EU[a] = V0` |
| `regret` | `m` arrays of `n` numbers | `R[a][θ] = max_a′ Q[a′][θ] − Q[a][θ]` |
| `expected_regret` | array of `m` numbers | `ER[a] = Σ_θ p[θ] · R[a][θ]` |
| `max_regret` | array of `m` numbers | `MR[a] = max_θ R[a][θ]` |
| `minimax_regret_actions` | array of indices | all `a` with `MR[a] = min_a′ MR[a′]` |
| `evpi` | number | `Σ_θ p[θ] · max_a Q[a][θ] − V0` |
| `value_with_information` | number | `Σ_z max_a Σ_θ p[θ] · L[z][θ] · Q[a][θ]`; only with an experiment |
| `evsi` | number | `value_with_information − V0`; only with an experiment |

`value_with_information` is in the joint form: the posterior is not formed and
nothing is divided by the probability of a signal, so a signal of probability zero
contributes zero. `max_regret` ranges over all `n` states, whatever their
probability. Ties are exact: an index set lists every action that attains the
optimum.

## What a checker should verify

1. The shapes agree, `prior` is a probability vector, and every column of
   `likelihood` is one.
2. Each claim equals the value its definition gives for the stated `payoff`, `prior`
   and `likelihood`.

The theorems that relate the claims (`evpi ≥ 0`, `evpi = min_a ER[a]`,
`bayes_actions = argmin ER`, `0 ≤ evsi ≤ evpi`) then hold of the certificate as
instances, and are checked for it in the package's test suite.

## Example

The two-state epidemic example with its 80/80 test:

```json
{
  "schema": "decision-certificate/1",
  "actions": ["do_nothing", "intervene"],
  "states": ["low", "high"],
  "payoff": [["0", "-100"], ["-30", "-30"]],
  "prior": ["3/5", "2/5"],
  "signals": ["negative", "positive"],
  "likelihood": [["4/5", "1/5"], ["1/5", "4/5"]],
  "claims": {
    "expected_utility": ["-40", "-30"],
    "act_now_value": "-30",
    "bayes_actions": [1],
    "regret": [["0", "70"], ["30", "0"]],
    "expected_regret": ["28", "18"],
    "max_regret": ["70", "30"],
    "minimax_regret_actions": [1],
    "evpi": "18",
    "value_with_information": "-106/5",
    "evsi": "44/5"
  }
}
```

## Requirements on the problem

The payoffs, the prior and the likelihood must have `Rational` or integer element
type; otherwise [`InexactCertificateError`](@ref) is thrown. The belief must be a
[`FiniteBelief`](@ref) or a [`PointBelief`](@ref).

## Schema `garbling-witness/1`

The proof that an experiment `e′` is a garbling of an experiment `e`, written by
`export_garbling_witness(io_or_path, e′, e)` from the matrix that
[`is_garbling_of`](@ref) finds (the witness of [`garbling_result`](@ref)).

| member | type | meaning |
|---|---|---|
| `schema` | string | always `"garbling-witness/1"` |
| `experiment` | `k` arrays of `n` numbers | the likelihood of `e`: `L[z][θ] = P(z \| θ)` |
| `garbled` | `k′` arrays of `n` numbers | the likelihood of `e′` |
| `garbling` | `k′` arrays of `k` numbers | `G[z′][z] = P(z′ \| z)`: rows are signals of `e′`, columns signals of `e` |

A checker verifies that the three matrices are rectangular with matching sizes, that
every column of `experiment` and of `garbling` is a probability vector, and that
`garbled[z′][θ] = Σ_z garbling[z′][z] · experiment[z][θ]` for every entry. It follows
that `e′` is worth no more than `e` in every decision problem with a single prior.

```json
{
  "schema": "garbling-witness/1",
  "experiment": [["9/10", "3/5", "1/5", "1/20"], ["1/10", "2/5", "4/5", "19/20"]],
  "garbled": [["37/50", "14/25", "8/25", "23/100"], ["13/50", "11/25", "17/25", "77/100"]],
  "garbling": [["4/5", "1/5"], ["1/5", "4/5"]]
}
```

[`WitnessError`](@ref) is thrown, and nothing is written, when `e′` is not a garbling
of `e`. The check is sound and not complete: a rejected witness does not show that no
other witness exists.

## Schema `e-admissible-witness/1`

The proof that an action is E-admissible, written by
`export_eadmissible_witness(io_or_path, problem, action)` from the solution of the
programme behind [`EAdmissible`](@ref). The same weights and prior are in
`decide(problem, EAdmissible()).details.weights` and `.witnesses`.

| member | type | meaning |
|---|---|---|
| `schema` | string | always `"e-admissible-witness/1"` |
| `payoff` | `m` arrays of `n` numbers | `Q[a][θ]`, the feasible actions |
| `priors` | `K` arrays of `n` numbers | the generating points of the credal set, one per row |
| `weights` | array of `K` numbers | convex weights on the generating points |
| `prior` | array of `n` numbers | the witness prior, `prior[θ] = Σ_k weights[k] · priors[k][θ]` |
| `action` | index | the action, counted from zero among the rows of `payoff` |

A checker verifies that every row of `priors` is a probability vector, that the
weights are nonnegative and sum to one, that `prior` is the stated mixture, and that
the action maximises `Σ_θ prior[θ] · Q[a][θ]` (ties allowed). The weights are needed,
not only the prior: they are what places the prior inside the credal set.

```json
{
  "schema": "e-admissible-witness/1",
  "payoff": [["1", "0"], ["0", "1"], ["3/5", "3/5"]],
  "priors": [["9/10", "1/10"], ["1/10", "9/10"]],
  "weights": ["1/2", "1/2"],
  "prior": ["1/2", "1/2"],
  "action": 2
}
```

Here the third action is best at neither generating point and is E-admissible only
through an interior prior. [`WitnessError`](@ref) is thrown, and nothing is written,
when the action is not E-admissible.

## Schema `not-e-admissible-witness/1`

Establishes that one action is **not** E-admissible over a credal set: no prior in the
convex hull of the generating points makes it a Bayes action. With it a "no" is
verified in exact arithmetic whatever found it, as a "yes" already is.

| member | type | meaning |
|---|---|---|
| `schema` | string | always `"not-e-admissible-witness/1"` |
| `payoff` | `m` arrays of `n` numbers | `Q[a][θ]`, rows = feasible actions, columns = states |
| `priors` | `K` arrays of `n` numbers | the generating points of the credal set, one per row |
| `action` | integer in `0 … m−1` | the action claimed not to be E-admissible |
| `mixed_action` | array of `m` numbers | `μ[b]`, a probability vector over the actions |
| `actions`, `states` | arrays of strings | labels, ignored by the checker |

**What a checker should verify.**

1. `payoff` is `m × n` with `m` the length of `mixed_action`; `priors` is `K × n`.
2. Every row of `priors` is nonnegative and sums to one.
3. `mixed_action` is nonnegative and sums to one, and `action < m`.
4. For **every** generating point `k`, strictly,
   `Σ_b μ[b] · E[b, k] > E[action, k]`, with `E[b, k] = Σ_θ priors[k][θ] · payoff[b][θ]`.

Strict dominance at every generating point is strict dominance at every prior of the
hull, so under every such prior some pure action is strictly better than `action`.
The witness is complete: every action that is not E-admissible has such a mixed
action (linear-programming duality; Lean `not_isEAdmissible_iff_isMixedDominated`).
The package finds it with the programme of [`mixed_dominance_model`](@ref), the dual
of the E-admissibility programme, and checks condition 4 exactly before writing.

```json
{
  "schema": "not-e-admissible-witness/1",
  "actions": ["left", "right", "hedge"],
  "states": ["s1", "s2"],
  "payoff": [["1", "0"], ["0", "1"], ["2/5", "2/5"]],
  "priors": [["1", "0"], ["0", "1"]],
  "action": 2,
  "mixed_action": ["1/2", "1/2", "0"]
}
```

Here the third action is maximal (no pure action beats it at both generating points)
and still not E-admissible: the even mixture of the other two beats it at both.
[`WitnessError`](@ref) is thrown, and nothing is written, when the action is
E-admissible.

## Schema `not-garbling-witness/1`

Establishes that the experiment `garbled` is **not** a garbling of `experiment`, by
exhibiting a decision problem in which deciding after `garbled` is worth strictly
more. The roles of the two members are those of `garbling-witness/1`: the positive
certificate says `garbled = G · experiment`; this one says that no such `G` exists.

| member | type | meaning |
|---|---|---|
| `schema` | string | always `"not-garbling-witness/1"` |
| `experiment` | `k` arrays of `n` numbers | `L[z][θ]`, the likelihood of `e` |
| `garbled` | `k′` arrays of `n` numbers | `L′[z′][θ]`, the likelihood of `e′` |
| `payoff` | `m ≥ 1` arrays of `n` numbers | the separating payoff table `Q[a][θ]` |
| `prior` | array of `n` numbers | the prior of the separating problem |

**What a checker should verify.**

1. Shapes: `experiment`, `garbled` and `payoff` have `n` columns; `prior` has `n`
   entries; `payoff` has at least one row.
2. Both likelihoods are column-stochastic; the prior is nonnegative and sums to one.
3. `value(garbled) > value(experiment)`, strictly, both recomputed as
   `Σ_z max_a Σ_θ prior[θ] · L[z][θ] · payoff[a][θ]`.

A garbling is never worth more, for any payoff table and any prior, so a strictly
larger value refutes it; no duality is involved in the check. The witness is
complete (Blackwell's theorem; Lean `not_isGarblingOf_iff_exists_separating`):
whenever `e′` is not a garbling of `e`, a separating table exists with one action
per signal of `e′` under any prior of full support. The package uses the uniform
prior, finds the table with the programme of [`separating_payoff_model`](@ref), and
checks condition 3 exactly before writing; any table that passes is a valid witness,
whatever rounding it went through.

```json
{
  "schema": "not-garbling-witness/1",
  "experiment": [["37/50", "14/25", "8/25", "23/100"], ["13/50", "11/25", "17/25", "77/100"]],
  "garbled": [["9/10", "3/5", "1/5", "1/20"], ["1/10", "2/5", "4/5", "19/20"]],
  "payoff": [["1", "1", "-1", "-1"], ["-1", "-1", "1", "1"]],
  "prior": ["1/4", "1/4", "1/4", "1/4"]
}
```

[`WitnessError`](@ref) is thrown, and nothing is written, when `e′` is a garbling of
`e`.

**Exact "no" answers.** With these two witnesses, a negative answer found by a
floating-point optimiser is reported as exact (`details.exact[action]` of an
`EAdmissible` result, `GarblingResult.exact`) whenever the rounded witness passes the
exact check, and as numerical otherwise.

## The fixture file

`test/fixtures/worked_examples.json` (schema `"decision-fixtures/1"`) holds the three
worked examples in the same number encoding. It is an object with a `description`
and an array `examples`; each example has `name`, `actions`, `states`, `baseline` (the
label of the no-intervention action), `payoff`, `prior`, optionally `credal` (an
array of priors), `experiments` (an object from experiment name to `signals`,
`likelihood`, and optionally `cost`, and `garbling` with `garbling_of` when the
experiment is a garbling `G[z′][z]` of another one in the same example), and
`expected`, the quoted figures: the claims above, with `value_with_information`,
`evsi` and `enbs` as objects keyed by experiment name, and `maximin`,
`maximin_actions`, `lower_expected_utility`, `gamma_maximin_actions` and
`e_admissible`.
