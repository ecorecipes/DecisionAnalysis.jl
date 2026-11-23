# Certificates

A certificate is a JSON document in exact rational arithmetic that an independent
checker can confirm without trusting this package. There are three kinds, each with
its own schema string:

| schema | written by | what it establishes |
|---|---|---|
| `decision-certificate/1` | [`export_decision_certificate`](@ref) | the numbers claimed for one decision problem |
| `garbling-witness/1` | [`export_garbling_witness`](@ref) | one experiment is a garbling of another |
| `e-admissible-witness/1` | [`export_eadmissible_witness`](@ref) | an action is E-admissible over a credal set |

The checker is the Lean executable in `proofs/`
(`cd proofs && lake exe check_certificate FILE.json`), which dispatches on `schema`
and exits with 0 when it accepts, 1 when it rejects and 2 when the file cannot be
decoded. This page is the definition of the formats; they are a contract, and a
change of meaning needs a new schema string.

The two witnesses come from linear programmes. The package solves those exactly for
rational problems, and checks each witness in exact arithmetic before writing it, so
an exported witness is never a rounded floating-point solution.

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
[`is_garbling_of`](@ref) finds.

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
