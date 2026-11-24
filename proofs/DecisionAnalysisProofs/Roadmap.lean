import DecisionAnalysisProofs.Basic

/-!
# Roadmap

Prose only: what is not proved, in the order in which the Julia package needs it, and what the
proved results ask of the package. This module has no declarations and no `sorry`; it is not
part of the default target.

## Julia behaviour that is tested or documented but not backed by a theorem

* **`total_variation` as a maximum over events**: the docstring's
  `TV(p, q) = max_A |p(A) − q(A)|` is not proved (only the half-ℓ¹ definition is used).
* **Attainment in the linear programmes.** `deficiency` and the upper end of `robust_evpi`
  are optima of programmes. The Lean statements are about feasible points
  (`IsDeficientWithin`) and about an optimum that is assumed to exist
  (`evpiEnvelope_optimum`); that the minimum or maximum is attained is not proved, and the
  simplex of `src/lp.jl` is not modelled.
* **S1a, the converse of the deficiency bound**: a value gap of at most `B δ` in every
  problem implies deficiency at most `δ`. Not proved. It would make `deficiency` certifiable
  from below by a decision problem, as `"not-garbling-witness/1"` certifies `δ > 0`.
* **T11, total variation under composition** (sup-TV is subadditive along composites and
  tensors): not proved, deliberately. The package has no function that takes a diagram of
  kernels with per-kernel errors; the statement would back nothing until one exists.
* **T13b** (commitment to a randomised rule restores monotonicity under Γ-maximin): not
  attempted; its statement has not been checked against a source.
* **`robust_evsi` in general.** `dilation_robust_evsi_negative` proves the test instance.
  `updatedLowerUtility` agrees with Julia only when every prior gives every signal positive
  probability; the case in which Julia drops priors is not modelled, and neither are
  `AlphaMaxmin` or `UpperExpectation`.
* `decide(...; atol, rtol)` with nonzero tolerances, `Satisfice`, `ParetoFront`, `Hurwicz`,
  `CVaR`, `enbs` and the robustness metrics have no Lean counterpart.

## Where the Julia text and the theorems differ

* **The docstring of `evpi` says that noise in the payoff table "biases EVPI upwards". That is
  false in general** (`evpi_bias_sign_indefinite`): EVPI is the difference of two
  upward-biased terms, and an instance has expected estimated EVPI `9/2` against a true EVPI
  of `5`. What is true: the act-now value, the perfect-information value and the value with
  information are each biased upwards (`actNowValue_meanTable_le`,
  `perfectValue_meanTable_le`, `valueWithInformation_meanTable_le`), and the chosen action is
  on average worth less than its reported score (`sum_mul_expectedUtility_meanTable_le`).
  The same applies to the report notes in `src/report.jl` ("the plain EVPI is biased upwards by
  the simulation noise", "maximising noisy cells overstates the plain EVPI"): provable for the
  perfect-information term, not for EVPI. For a sampled **belief** with exact payoffs the
  direction is the opposite and is proved: EVPI is biased downwards
  (`sum_mul_evpi_le_evpi_mixture`), because only the act-now term is overstated. "Lowers
  every value of information" is proved for EVPI only; for EVSI and EVPPI both terms are
  convex in the prior.
* **`decision_loss_bound(B, ε)`: the docstring takes `B` to be the range of the whole table.**
  The theorem needs only the largest range of a single action's payoffs across states (row
  range), which is never larger (`IsBayes.decision_loss_le`). The constant `2` is tight
  (`decision_loss_bound_tight`). The beliefs need the same total mass and nothing else.
* **`deficiency`: the constant `B δ` in the docstring is right** (not `2 B δ`), and again `B`
  can be smaller than the table range: the largest spread of payoffs **within a state**, which
  is the largest entry of the regret table (`IsDeficientWithin.valueWithInformation_le`). The
  bound needs a probability prior and both experiments normalised; it needs nothing else of
  the experiments.
* **Blackwell's converse needs a prior of full support and at least one signal of `e′`**
  (`isGarblingOf_of_forall_valueWithInformation_le`). A separating problem built with a prior
  that has a zero entry can fail to exist; the exporter must use a full-support prior.
* **An empty credal set**: no action is E-admissible (`IsEAdmissible` needs weights summing
  to one), so the not-E-admissible checker accepts a document with no extreme points. Julia's
  `CredalSet` cannot be empty; nothing to change, but the two sides agree only because of
  that constructor check.

## Consequences for the package

Each line names a proved result and the Julia function or check it motivates.

* `Certificate.RawNotEAdmissible.accepts_sound`, `not_isEAdmissible_iff_isMixedDominated`
  (T12b): every action that `decide(problem, EAdmissible())` leaves out has a checkable
  witness, the dual solution of `eadmissibility_model`. **Done in Julia**: `export_not_eadmissible_witness`
  (`"not-e-admissible-witness/1"`), and a "no" from a floating-point optimiser is marked exact
  when the rounded witness passes the exact check.
* `Certificate.RawNotGarbling.accepts_sound`, `not_isGarblingOf_iff_exists_separating` (S1):
  every `nothing` of `is_garbling_of` has a checkable witness, a decision problem with one
  action per signal of `e′` and the uniform prior. **Done in Julia**: `export_not_garbling_witness`
  (`"not-garbling-witness/1"`), from the programme `separating_payoff_model`, and
  `GarblingResult.separating` carries the separating problem as a `DecisionProblem`, which
  is also the most useful thing to show a user who asks why two experiments are not ordered.
* `isMixedDominated_of_not_isMaximal`: an action rejected by `Maximal()` needs no programme
  to certify that it is not E-admissible; the pure dominating action is the witness. **Done
  in Julia**: `decide(problem, EAdmissible())` solves no programme for such an action and
  names the dominating action in `details.dominated_by`.
* `evpi_bias_sign_indefinite`, `sum_mul_expectedUtility_meanTable_le` (T16): **to do in
  Julia**: correct the docstring of `evpi`; if a selection-bias note is added to
  `DecisionReport`, attach it to the reported best value and the chosen action, not to EVPI or
  EVSI.
* `IsBayes.decision_loss_le`, `IsDeficientWithin.valueWithInformation_le`: **done in
  Julia**: `largest_action_range` (for `decision_loss_bound`, which also takes a table or a
  problem) and `largest_regret` (for the deficiency bound, `deficiency_value_bound`) return
  the two sharper ranges, beside `payoff_range`, the table range that users would otherwise
  pass.
* `evpiEnvelope_optimum`, `evpi_max_not_at_generating_point` (T15): the upper end of the
  envelope needs the programme and the lower end does not; `robust_evpi` is right on both.
  The optimal `λ` is the prior at which EVPI is largest; `RobustVOI.upper_prior` returns
  it.
* `dilation_robust_evsi_negative` (T13) and `valueWithInformation_garble_le` (T6, single
  prior only): `robust_evsi` must not assume monotonicity or nonnegativity, and must keep
  returning negative values unclamped.
* `minimaxRegret_violates_iia` (T8): the minimax-regret choice depends on the menu of
  actions, including actions that are never chosen. Done in Julia: the rule's result carries
  the note (`details.note`) and `DecisionReport` says so; `feasible_actions` filters change
  that choice.
* `evsi_eq_zero_iff` and `evsi_eq_zero_iff_of_pos` (T14): an experiment has zero EVSI exactly
  when one action is optimal after every signal of positive probability. Done in Julia:
  `information_can_change_decision(problem, experiment)`, held to `evsi > 0` by a property
  test.
* `signalValue_eq_zero` and `valueWithInformation_eq_posterior_form`: the joint form used by
  `value_with_information` needs no guard for signals of probability zero, and agrees with
  the posterior form; `bayes_invert` is right to return `nothing` for such signals, and no
  caller should need the posterior to compute a value.
* `evsi_le_evpi` needs the experiment to be on the same states as the payoff table and the
  prior to be nonnegative; `evsi_nonneg` needs the experiment to be normalised. The checks in
  `Experiment` and `_check_states` are therefore load-bearing, not cosmetic.
* `IsEAdmissible` quantifies over the convex hull, and `Examples.flood_isEAdmissible` rejects
  "relocate" through `IsEAdmissible.isMaximal`. The witness checker needs the hull **weights**
  as well as the witness prior. Done in Julia: `details.weights` carries them and
  `export_eadmissible_witness` writes them.
* `IsGarblingOf.trans` and `Certificate.RawGarbling.accepts_sound`: the matrix returned by
  `is_garbling_of` is a complete, independently checkable certificate. Done in Julia:
  `export_garbling_witness` writes schema `"garbling-witness/1"`.
* `bayesAction_rescale` and `isMinimaxRegret_rescale`: `first_choice` is invariant under a
  change of payoff units for both rules, so reports need not record the units to be
  reproducible.
* `evppi_mono_of_factorsThrough`: `evppi` is monotone only between partitions one of which
  refines the other; two unrelated partitions are not ordered and the report should not rank
  them as if they were.

## External libraries

`gametheoryinlean/EconCSLib` (Apache-2.0, Lean v4.30.0, Mathlib at the same commit as this
project) has Farkas' lemma, linear-programming duality and the minimax theorem over ordered
fields, and was tried as a dependency for T12b and S1. Its licence and pins are compatible,
but its modules log deprecation warnings at this toolchain, and `lake build --wfail` fails on
warnings in dependencies. The one statement needed (Ville's alternative) is therefore proved
here, in `Finite/Farkas.lean`, in about 150 lines; the project has no dependency beyond
Mathlib, mdgen and `finite_kernels_proofs`.
-/
