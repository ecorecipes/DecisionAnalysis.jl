# Expected outcomes of `check_certificate` on the fixtures

Run from `proofs/`: `make check-fixtures` (or `lake exe check_certificate FILE.json`).
The exit code is 0 when the proved checker accepts, 1 when it rejects, 2 when the file cannot
be decoded. The prefix of the file name fixes the expected code, and
`scripts/check_fixtures.sh` enforces it.

## Written by Julia

Exported with `export_decision_certificate` from the worked examples and committed unchanged.

| File | Schema | Expected |
|:--|:--|:--|
| `flood.json` | `decision-certificate/1` | accepted (exit 0) |
| `flood_gauge.json` | `decision-certificate/1`, with the river gauge | accepted (exit 0) |
| `equipment_test.json` | `decision-certificate/1`, with the 80/90 test | accepted (exit 0) |

## Written by hand

Each is `flood_gauge.json` with one change.

| File | Change | Expected |
|:--|:--|:--|
| `reject_wrong_evpi.json` | `evpi` is `10`, not `207/20` | rejected (exit 1): `evpi: FAIL` |
| `reject_wrong_bayes_actions.json` | `bayes_actions` is `[0, 1]` | rejected (exit 1) |
| `reject_wrong_evsi.json` | `evsi` is `49/25` (the garbled gauge's) | rejected (exit 1) |
| `reject_prior_not_normalised.json` | prior sums to `21/20` | rejected (exit 1) |
| `reject_negative_prior.json` | prior has `-1/20` and still sums to one | rejected (exit 1) |
| `reject_likelihood_not_stochastic.json` | last likelihood column sums to `19/20` | rejected (exit 1) |
| `malformed_ragged_payoff.json` | second payoff row has three entries | malformed (exit 2) |
| `malformed_decimal_number.json` | `act_now_value` is `"-19.5"` | malformed (exit 2) |

## Witnesses (written by `export_garbling_witness` and `export_eadmissible_witness`)

| File | Content | Expected |
|:--|:--|:--|
| `garbling_flood.json` | gauge, garbled gauge, and the flip matrix `G[z'][z]` | accepted (exit 0) |
| `reject_garbling_wrong_product.json` | a stochastic `G` whose product is not the garbled gauge | rejected (exit 1) |
| `reject_garbling_not_stochastic.json` | a `G` with a column summing to `11/10` | rejected (exit 1) |
| `eadmissible_flood_raise_levee.json` | "raise levee" is Bayes under ensemble member 1 (weights `0, 1, 0`) | accepted (exit 0) |
| `eadmissible_flood_interior_tie.json` | "do nothing" is Bayes (tied) under the mixture `1/2, 1/2, 0` | accepted (exit 0) |
| `reject_eadmissible_relocate.json` | "relocate" is not Bayes under member 1 | rejected (exit 1) |
| `reject_eadmissible_negative_weight.json` | weights `1/2, 1, -1/2` | rejected (exit 1) |
| `reject_eadmissible_prior_not_mixture.json` | the stated prior is not the stated mixture | rejected (exit 1) |

## Certified "no" answers (written by hand; schemas in `../SCHEMAS-PROPOSED.md`)

No Julia exporter writes these yet.

| File | Content | Expected |
|:--|:--|:--|
| `not_eadmissible_mixed.json` | the "hedge" instance of `test/test_credal.jl`: payoffs `(1,0), (0,1), (2/5,2/5)`, the two point masses; the hedge is maximal but the mixture `1/2, 1/2, 0` beats it at both | accepted (exit 0) |
| `not_eadmissible_flood_relocate.json` | "relocate" is beaten by "do nothing" (mixed action `1, 0, 0`) under every ensemble member | accepted (exit 0) |
| `reject_not_eadmissible_pure.json` | as `not_eadmissible_mixed.json` with the pure action `1, 0, 0`, which loses at the second extreme point | rejected (exit 1) |
| `reject_not_eadmissible_not_normalised.json` | mixed action `1, 1, 0` | rejected (exit 1) |
| `reject_not_eadmissible_raise_levee.json` | "raise levee" is E-admissible; "do nothing" does not beat it under member 1 | rejected (exit 1) |
| `reject_not_eadmissible_tie.json` | payoffs `(2,2), (0,4), (4,0)`: the mixture `0, 1/2, 1/2` only ties the first action, which is E-admissible | rejected (exit 1) |
| `not_garbling_flood.json` | the gauge is not a garbling of the garbled gauge: the flood problem values it strictly higher | accepted (exit 0) |
| `reject_not_garbling_is_garbling.json` | the two experiments swapped: the garbled gauge is a garbling of the gauge | rejected (exit 1) |
| `reject_not_garbling_same_experiment.json` | the gauge against itself: equal values, not strictly larger | rejected (exit 1) |
| `reject_not_garbling_prior_not_normalised.json` | prior sums to `21/20` | rejected (exit 1) |
| `malformed_not_garbling_no_prior.json` | the `prior` member is missing | malformed (exit 2) |
