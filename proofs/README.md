# DecisionAnalysisProofs

Lean 4 / Mathlib formalisation for `DecisionAnalysis.jl` (toolchain v4.30.0, Mathlib v4.30.0).

**Purpose.** These proofs back the Julia code and its certificates. They give each exported
quantity a definition that Julia's exact (`Rational`) results are compared with, prove the
properties that the Julia test suite asserts on random problems, and prove sound the checkers
that accept or reject what Julia exports. They are not a development of decision theory: a
result is here because a Julia function, a property test, a certificate or an API decision
rests on it.

```sh
cd proofs
lake exe cache get    # prebuilt Mathlib oleans (once per Mathlib revision)
make build            # lake build --wfail: the default target, sorry-free
make audit            # #print axioms for every headline theorem, and a source scan
make check-fixtures   # check_certificate on certificate-fixtures/, exit codes as in EXPECTED.md
make roadmap          # builds the remaining-work module (prose only)
```

The dependency checkout is shared with the other ecosystem `proofs/` projects via `packagesDir`
(`../../.lake-packages`). The project depends by path on `../../FiniteKernels.jl/proofs`
(`finite_kernels_proofs`). CI (`.github/workflows/lean.yml`) runs `lake build --wfail`, the
audit and the fixture check. `Roadmap.lean` is not imported by the default target.

## Documents

The readable documents of the sibling projects (`*Proofs.md`, `.html`, `.pdf`, from mdgen and
pandoc) are not generated for this project yet; `make mdgen` writes one Markdown file per
module into `.mdgen_tmp/`. The module docstrings are written for that pipeline.

## What is formalised

One model serves both the theorems and the computations: finite types of actions, states and
signals, a payoff table `u : A → Θ → K`, a prior `p : Θ → K`, an experiment `e : Θ → Z → K`
with `e θ z = P(z | θ)` (Julia's `likelihood[z, θ]`), over any linearly ordered field `K`.
The theorems hold for every such `K`, in particular `ℝ`; the worked examples and the
certificate checker instantiate the same definitions at `K = ℚ`. There is no separate
"computable" copy of the definitions to relate to the "real" one.

### By Julia function

| Julia | Lean definition | What is proved about it |
|:--|:--|:--|
| `act_now_value`; `decide(p, MaxExpectedUtility())` | `expectedUtility`, `actNowValue`, `IsBayes`, `maximisers` | attained (`exists_isBayes`); the chosen set is exactly the optimal actions (`mem_maximisers`) |
| `first_choice` | `firstArgmax`, `firstArgmin`, `bayesAction`, `minimaxRegretAction`, `gammaMaximinAction` | optimal (`firstArgmax_spec`), first among ties (`lt_of_lt_firstArgmax`), determined by those two properties (`eq_firstArgmax`), depends only on the ranking (`firstArgmax_congr_of_order`), unchanged by a positive rescaling with a state-dependent shift (`bayesAction_rescale`) |
| `value_with_information`, `evsi` | `valueWithInformation`, `evsi` (joint form, no division) | equals the posterior form `Σ_z P(z) max_a E[u | z]` (`valueWithInformation_eq_posterior_form`); zero-probability signals contribute zero (`signalValue_eq_zero`) |
| `information_can_change_decision(problem, experiment)` | `¬ ∃ a, ∀ z, 0 < signalProbability p e z → ∀ a', signalValue u p e z a' ≤ signalValue u p e z a` | equivalent to `evsi > 0` (`evsi_eq_zero_iff`, `evsi_eq_zero_iff_of_pos`, with `evsi_nonneg`) |
| `evpi` | `perfectValue`, `evpi` | `evpi_nonneg`, `minOver_expectedRegret_eq_evpi` |
| `evppi(problem, partition)` | `partialValue`, `evppi` | equals the EVSI of the deterministic experiment `θ ↦ cell(θ)` (`evppi_eq_evsi`) |
| `regret_table`, `expected_regret`, `max_regret`; `decide(p, MinimaxRegret())` | `regret`, `expectedRegret`, `maxRegret`, `IsMinimaxRegret` | `expectedRegret_eq`, `regret_nonneg` |
| `bayes_invert` | `signalProbability`, `posterior` | normalised when `P(z) ≠ 0` (`posterior_normalised`), nonnegative, marginals sum to one, posteriors average to the prior |
| `garble`, `is_garbling_of`, `garbling_result` (programme: `garbling_model`) | `garble`, `IsGarblingOf` | `garble` is `Kernel.comp` of FiniteKernels (`garble_eq_comp`); preorder (`IsGarblingOf.refl`, `.trans`) |
| `decide(p, GammaMaximin())`, `EAdmissible()` (programme: `eadmissibility_model`), `Maximal()`, `IntervalDominance()` | `IsGammaMaximin`, `IsEAdmissible` (over the convex hull), `IsMaximal`, `IsIntervalUndominated`, `IsMixedDominated` | inclusions below; a maximal action is not strictly beaten by one rival at every prior of the hull (`IsMaximal.not_dominated_on_hull`); an action is not E-admissible iff a mixed action strictly beats it at every extreme point (`not_isEAdmissible_iff_isMixedDominated`, T12b) |
| `total_variation`, `decision_loss_bound(B, ε)` | `totalVariation` (half the ℓ¹ distance) | each action's expected utility moves by at most `B · TV` (`abs_expectedUtility_sub_le`); a Bayes action under `q` loses at most `2 · B · TV(p, q)` under `p` (`IsBayes.decision_loss_le`, T10), `+ ε′` for an `ε′`-optimal choice (`decision_loss_le`); the constant 2 is attained (`decision_loss_bound_tight`); `0 ≤ TV ≤ 1` |
| `deficiency(e, e′)` (programme: `deficiency_model`) | `IsDeficientWithin e e' δ` (a feasible point of the programme with objective `δ`) | value(e′) ≤ value(e) + `B · δ` for every probability prior (`IsDeficientWithin.valueWithInformation_le`, T6a); `δ = 0` iff garbling (`isDeficientWithin_zero_iff`) |
| `is_garbling_of`, `garbling_result`: the answer "no" | `IsGarblingOf` | Blackwell's converse: not a garbling iff some decision problem values `e′` strictly higher (`not_isGarblingOf_iff_exists_separating`, `isGarblingOf_iff_forall_valueWithInformation_le`, S1) |
| `robust_evpi`: `lower`, `upper` (programme: `evpi_envelope_model`) | `mixture`, `EvpiEnvelopeFeasible`, `evpiEnvelopeObjective` | EVPI is concave in the prior (`sum_mul_evpi_le_evpi_mixture`); its minimum over the hull is at a generating point (`minOver_evpi_le_evpi_mixture`); an optimum of the programme is the EVPI at `Σ λ_k p_k` and the largest over the hull (`evpiEnvelope_optimum`); the maximum need not be at a generating point (`evpi_max_not_at_generating_point`) (T15) |
| `robust_evsi` under `GammaMaximin` | `updatedLowerUtility`, `strategyPayoff`, `strategyLowerValue` | the dilation instance: `act_now = 1/10`, `informed = 0`, each member's EVSI `11/10` (`dilation_robust_evsi_negative`, T13) |
| a payoff table estimated from simulator replicates (docstring of `evpi`) | `meanTable` | reported act-now, perfect and informed values are biased upwards (`actNowValue_meanTable_le`, `perfectValue_meanTable_le`, `valueWithInformation_meanTable_le`); the chosen action disappoints on average (`sum_mul_expectedUtility_meanTable_le`); the bias of EVPI has no sign (`evpi_bias_sign_indefinite`) (T16) |

### By Julia property test

| Julia test (`test/`) | Lean theorem |
|:--|:--|
| `test_voi.jl`: `evpi(problem) >= 0` | `evpi_nonneg` (T1) |
| `test_voi.jl`: `evpi(problem) == minimum(er)` | `minOver_expectedRegret_eq_evpi` (T2) |
| `test_voi.jl`: `argmin_set(er) == argmax_set(eu)`; regret and utility differ by a constant | `isBayes_iff_expectedRegret_le`, `setOf_isBayes_eq` (T3), `expectedRegret_eq` |
| `test_voi.jl`: `0 <= evsi <= evpi`; identity experiment gives EVPI; constant experiment gives 0 | `evsi_nonneg`, `evsi_le_evpi` (T5), `evsi_identity`, `evsi_constant` |
| `test_voi.jl`: garbling never raises EVSI | `valueWithInformation_garble_le`, `IsGarblingOf.evsi_le` (T6) |
| `test_voi.jl`: EVPPI zero for one cell, EVPI for single states, monotone under refinement, equal to the EVSI of the revealing experiment | `evppi_const`, `evppi_id`, `evppi_nonneg`, `evppi_le_evpi`, `evppi_mono`, `evppi_mono_of_factorsThrough` (T4), `evppi_eq_evsi` |
| `test_voi.jl`: positive affine maps keep the Bayes and minimax-regret actions and scale EVPI, EVSI and regret | `isBayes_rescale`, `evpi_rescale`, `evsi_rescale`, `evppi_rescale`, `regret_rescale`, `isMinimaxRegret_rescale` (T7) |
| `test_voi.jl`: signals of probability zero contribute nothing | `signalValue_eq_zero`, `maxOver_signalValue_eq_zero` |
| `test_experiments.jl`: Bayesian inversion (marginal sums to one, posteriors average to the prior) | `sum_signalProbability`, `posterior_normalised`, `sum_signalProbability_mul_posterior` |
| `test_experiments.jl`, `test_programs.jl`: `is_garbling_of` / `garbling_result` witness (`G * L == L′`, `G` stochastic), by each solver route | `Certificate.GarblingWitnessOk.isGarblingOf`, `Certificate.RawGarbling.accepts_sound` |
| `test_credal.jl`: Γ-maximin value below every member's expected utility | `IsGammaMaximin.gammaMaximinValue_le`, `gammaMaximinValue_le_actNowValue` (T12) |
| `test_credal.jl`: member Bayes ⊆ E-admissible ⊆ maximal ⊆ interval dominance; Γ-maximin choices are maximal | `IsBayes.isEAdmissible`, `IsEAdmissible.isMaximal`, `IsMaximal.isIntervalUndominated`, `IsGammaMaximin.isMaximal` (T12) |
| `test_credal.jl`, `test_programs.jl`: every E-admissibility witness is a prior in the set under which the action is Bayes, by each solver route | `Certificate.EAdmissibleWitnessOk.isEAdmissible` |
| `test_lean_mirror.jl`: a maximal action is not strictly beaten by one rival at every prior of the hull | `IsMaximal.not_dominated_on_hull` (T12) |
| `test_rules.jl`: ties, all returned and first in table order chosen | `mem_maximisers`, `firstArgmax_mem_maximisers`, `lt_of_lt_firstArgmax`, `eq_firstArgmax` |
| `test_lean_mirror.jl`: `first_choice` depends only on the ranking of the scores, in any ordered type | `firstArgmax_congr_of_order` |
| `test_lean_mirror.jl`: the first Bayes action is unchanged by `s * Q[a, θ] + b[θ]` with `s > 0` | `bayesAction_rescale`, `expectedUtility_rescale` (T7) |
| `test_rules.jl`: minimax regret violates independence of irrelevant alternatives; `test_api.jl`: the result of `MinimaxRegret` and the report carry the warning | `minimaxRegret_violates_iia` (T8; a different instance from the Julia test's) |
| `test_api.jl`: `information_can_change_decision(p, e) == (evsi(p, e) > 0)` on random rational problems | `evsi_eq_zero_iff`, `evsi_eq_zero_iff_of_pos` (T14) |
| `test_voi.jl`: `0 <= loss <= decision_loss_bound(B, total_variation(p, q))` | `decision_loss_nonneg`, `IsBayes.decision_loss_le` (T10) |
| `test_voi.jl`: `total_variation` values | `totalVariation` (definition), `totalVariation_le_one` |
| `test_experiments.jl`, `test_programs.jl`: a witness exactly when `deficiency` is zero (`is_garbling == iszero(deficiency(e, f))`) | `isDeficientWithin_zero_iff` |
| `test_voi.jl`: `v.lower == minimum(members)`, `v.lower <= minimum(grid)`, `v.upper >= maximum(members)`, `v.upper >= maximum(grid)` for `robust_evpi` | `minOver_evpi_le_evpi_mixture`, `evpiEnvelope_optimum` (T15) |
| `test_voi.jl`: "robust EVSI can be negative and is not clamped" (`act_now == 1//10`, `informed == 0`, `value == -1//10`, members positive) | `dilation_robust_evsi_negative` (T13) |
| `test_credal.jl`: "a hedge that is never best: maximal but not E-admissible" | fixture `not_eadmissible_mixed.json` (the same instance) with `Certificate.RawNotEAdmissible.accepts_sound`; in general `IsMixedDominated.not_isEAdmissible` |
| `test_fixtures.jl`: the three worked examples | `Finite/Examples.lean` (T9), every figure |
| `test_certificates.jl`: exported claims recomputed | `Certificate.check_sound`, `Certificate.certificate_spec` |

Tested or documented in Julia but **not** proved here: that the programmes of `deficiency`
and `robust_evpi` attain their optima, the identity `TV = max_A |p(A) − q(A)|`, and
`robust_evsi` beyond the test instance; see `Roadmap.lean`. The linear programmes themselves
(`eadmissibility_model`, `garbling_model`, `deficiency_model`, `evpi_envelope_model`, solved
by the built-in simplex or by a JuMP optimiser) are not modelled as algorithms: whatever
solver produced it, a witness is checked exactly in Julia and then by the checkers below,
whose soundness is what is proved. For `deficiency_model` and `evpi_envelope_model` the
theorems are about the feasible set and objective written in their docstrings.

**Where the Julia text and the theorems differ** (details in `Roadmap.lean`): the docstring
of `evpi` says noise in the table "biases EVPI upwards", which is false in general
(`evpi_bias_sign_indefinite`); the `B` of `decision_loss_bound` and of the deficiency bound
can each be smaller than the table range (largest row range, and largest regret,
respectively); the constant `B δ` for `deficiency` is correct.

### By certificate

`lake exe check_certificate FILE.json` dispatches on the document's `"schema"`.

| Schema | Written by | Checker | Soundness theorem |
|:--|:--|:--|:--|
| `decision-certificate/1` | `export_decision_certificate` | `Certificate.check` | `Certificate.certificate_spec`: every claim equals the verified definition; `bayes_actions` and `minimax_regret_actions` list exactly the optimal actions; `evpi ≥ 0` and is the minimum expected regret; `0 ≤ evsi ≤ evpi` |
| `garbling-witness/1` | `export_garbling_witness`: the matrix returned by `is_garbling_of` (programme `garbling_model`, any `optimizer`; verified exactly before writing) | `Certificate.checkGarbling` | `Certificate.RawGarbling.accepts_sound`: `IsGarblingOf e′ e`, hence value(e′) ≤ value(e) for every problem |
| `e-admissible-witness/1` | `export_eadmissible_witness`: weights and prior from the E-admissibility programme (`eadmissibility_model`, any `optimizer`; verified exactly before writing) | `Certificate.checkEAdmissible` | `Certificate.RawEAdmissible.accepts_sound`: `IsEAdmissible`, hence `IsMaximal` |
| `not-e-admissible-witness/1` (proposed) | no exporter yet; the dual solution of `eadmissibility_model` | `Certificate.checkNotEAdmissible` | `Certificate.RawNotEAdmissible.accepts_sound`: `¬ IsEAdmissible`. Complete: `not_isEAdmissible_iff_isMixedDominated` |
| `not-garbling-witness/1` (proposed) | no exporter yet; the Farkas multipliers of `garbling_model` | `Certificate.checkNotGarbling` | `Certificate.RawNotGarbling.accepts_sound`: `¬ IsGarblingOf e′ e`. Complete: `not_isGarblingOf_iff_exists_separating` |

Numbers are JSON strings `"n/d"` or `"n"`; indices are JSON integers from zero; payoff rows are
actions; likelihood rows are signals; `garbling[z′][z] = P(z′ | z)`. The witness schemas are

```json
{"schema": "garbling-witness/1", "experiment": [[...]], "garbled": [[...]], "garbling": [[...]]}
{"schema": "e-admissible-witness/1", "payoff": [[...]], "priors": [[...]], "weights": [...],
 "prior": [...], "action": 0}
{"schema": "not-e-admissible-witness/1", "payoff": [[...]], "priors": [[...]], "action": 0,
 "mixed_action": [...]}
{"schema": "not-garbling-witness/1", "experiment": [[...]], "garbled": [[...]],
 "payoff": [[...]], "prior": [...]}
```

The last two certify **negative** answers and are specified in `SCHEMAS-PROPOSED.md`, with
where the Julia side finds each witness; they are not yet in `docs/src/certificates.md`.

The checker recomputes each claim with the definitions of `Finite/Problem.lean` at `K = ℚ`
and prints one line per claim. Exit code 0: accepted; 1: rejected; 2: cannot be decoded.
Fixtures and their expected outcomes are in `certificate-fixtures/EXPECTED.md`; three of them
were written by the Julia package itself, and those for the two proposed schemas by hand.

### Modules

| Module | Contents |
|:--|:--|
| `Basic.lean` | Introduction, the Julia-to-Lean table, the smoke theorem |
| `Finite/Extrema.lean` | `maxOver`, `minOver` (attained), affine and sum lemmas, evaluation on `Fin 2`, `3`, `4` |
| `Finite/Problem.lean` | The definitions: expected utility, values, EVPI, EVPPI, EVSI, regret |
| `Finite/Regret.lean` | T2, T3 |
| `Finite/Affine.lean` | T7 |
| `Finite/Garbling.lean` | T0, T6 |
| `Finite/ValueOfInformation.lean` | T1, T4, T5, T14 |
| `Finite/Posterior.lean` | `bayes_invert`; joint form equals posterior form |
| `Finite/Credal.lean` | T12 |
| `Finite/Selection.lean` | Chosen sets and `first_choice` |
| `Finite/MinimaxRegretIIA.lean` | T8 |
| `Finite/Examples.lean` | T9 |
| `Finite/Fidelity.lean` | T10: `totalVariation`, the `2Bε` bound and its tightness |
| `Finite/Deficiency.lean` | T6a: deficiency bounds the value gap by `B δ` |
| `Finite/SelectionBias.lean` | T16: bias from maximising estimated payoffs |
| `Finite/Envelope.lean` | T15: the EVPI envelope and the programme of `evpi_envelope_model` |
| `Finite/Farkas.lean` | Farkas' lemma and Ville's alternative over an ordered field |
| `Finite/Alternative.lean` | T12b and S1: completeness of the "no" certificates |
| `Finite/Dilation.lean` | T13: the negative robust EVSI of the Julia test |
| `Theory/FiniteKernels.lean` | The link to `FiniteKernelsProofs` kernels at `K = ℝ` |
| `Certificate/Check.lean` | The decision-certificate checker and its soundness |
| `Certificate/Witness.lean` | The garbling and E-admissibility witness checkers and their soundness |
| `Certificate/Refutation.lean` | The not-E-admissible and not-a-garbling witness checkers and their soundness |
| `Roadmap.lean` | What remains; consequences for the package (prose only) |
| `CheckCertificate.lean` | The executable: JSON decoding and printing (trusted) |

## Consequences for the package

Proved results that suggest or constrain functionality (the full list is in `Roadmap.lean`):

* `minimaxRegret_violates_iia`: the minimax-regret choice depends on the menu, including on
  actions that are never chosen, and feasibility filters change the choice. Done in the
  package: the result of `MinimaxRegret` carries the warning (`details.note`) and
  `DecisionReport` repeats it.
* `evsi_eq_zero_iff`: EVSI is zero exactly when one action is optimal after every signal.
  Done in the package: `information_can_change_decision(problem, experiment)` is that check,
  and `DecisionReport` says of an experiment that fails it that it cannot change the
  decision.
* `signalValue_eq_zero`, `valueWithInformation_eq_posterior_form`: `value_with_information`
  needs no guard for zero-probability signals, and `bayes_invert` is right to return
  `nothing` for them.
* `Certificate.RawEAdmissible.accepts_sound` needs the hull weights. Done in the package:
  `details.weights` carries them and `export_eadmissible_witness` writes them.
* T6 is for a single prior; `robust_evsi` must not assume it
  (`dilation_robust_evsi_negative`).
* `not_isEAdmissible_iff_isMixedDominated`, `not_isGarblingOf_iff_exists_separating`: every
  "no" of `decide(problem, EAdmissible())` and of `is_garbling_of` has a witness that can be
  checked exactly, so a "no" from a floating-point optimiser need not stay numerical. To do
  in the package: the two exporters of `SCHEMAS-PROPOSED.md`.
* `evpi_bias_sign_indefinite`: the docstring of `evpi` should not say that noise biases EVPI
  upwards; a selection-bias note belongs on the reported best value and the chosen action.
* `IsBayes.decision_loss_le`, `IsDeficientWithin.valueWithInformation_le`: both bounds hold
  with a smaller `B` than the table range; helpers for the two sharper ranges would be cheap.

## Design notes

* **Scope.** Exact finite-model results, not a verification of Julia's floating-point arrays.
  The Lean and Julia definitions are separate hand-written texts, checked to agree on exact
  problems by the certificate checker and the shared fixtures. No theorem says that the Julia
  code executes these definitions.
* **Trusted base of the checker.** `Lean.Json.parse`, the decoding of JSON into `Raw`
  structures and the printing in `CheckCertificate.lean` are not proved. The verdict is the
  value of `Certificate.check` (or `checkGarbling`, `checkEAdmissible`), about which the
  soundness theorems are stated.
* **Witnesses in both directions.** A rejected witness says nothing about whether another
  exists; a negative answer has its own witness instead. Each of the four checkers is sound,
  and the two pairs are complete: an action is E-admissible or has a dominating mixed action
  (`isEAdmissible_or_isMixedDominated`), and `e′` is a garbling of `e` or some decision
  problem separates them (`not_isGarblingOf_iff_exists_separating`). Soundness of the
  checkers does not depend on the duality argument (`Finite/Farkas.lean`); only completeness
  does.
* **No dependency for duality.** EconCSLib (Apache-2.0, same Mathlib commit) was tried for
  Farkas and minimax; it is compatible, but its modules log deprecation warnings and
  `lake build --wfail` fails on them, so Ville's alternative is proved in
  `Finite/Farkas.lean`.
* **Tolerances.** `decide` with nonzero `atol` or `rtol` is not modelled; certificates are
  exported with both set to zero.
* **Weakest hypotheses.** T6 needs only the garbling kernel to be stochastic; T2, T3 and T7
  need nothing of the prior; T7 allows a state-dependent shift. Hypotheses are stated where
  used, so a Julia guard that the theorems do not need is visible as such.
* `set_option linter.unusedSectionVars false` is set in the modules that declare one
  ordered-field context for the whole file.
