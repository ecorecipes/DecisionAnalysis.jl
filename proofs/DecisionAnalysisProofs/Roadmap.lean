import DecisionAnalysisProofs.Basic

/-!
# Roadmap

Prose only: what is not proved, in the order in which the Julia package needs it. This module
has no declarations and no `sorry`; it is not part of the default target.

## Julia behaviour that is tested or documented but not yet backed by a theorem

* **T10, decision fidelity** (`decision_loss_bound`, `total_variation`, `src/fidelity.jl`;
  asserted in `test/test_voi.jl`, "properties on random problems, exact"): deciding under a
  wrong belief `q` loses at most `2 · B · TV(p, q)`, with `B` the payoff range and `TV` half
  the ℓ¹ distance. Not proved here. The constant depends on those two conventions and they
  must be fixed in the definitions before the statement is written.
* **T6a, deficiency bounds the value gap** (`deficiency`, `src/experiments.jl`; the docstring
  states that the value of `e′` exceeds that of `e` by at most `B · δ(e, e′)`): not proved.
  Needs T10 and T11.
* **T11, total variation under composition** (sup-TV is subadditive along composites and
  tensors): not proved.
* **T13, free information can hurt under Γ-maximin** (`robust_evsi` "can be negative and is
  not clamped", `test/test_voi.jl`): the Julia test instance is not yet a Lean theorem.
* **T15, EVPI is concave in the prior** (`robust_evpi`: minimum of the envelope at an extreme
  point of the credal set): not proved.
* **T16, optimiser's curse** (selection-bias note in the report): not proved.
* `decide(...; atol, rtol)` with nonzero tolerances, `Satisfice`, `ParetoFront`, `Hurwicz`,
  `CVaR`, `enbs` and the robustness metrics have no Lean counterpart.

## Completeness of the witness checkers

* **S1, Blackwell converse**: value dominance for every problem implies garbling. Until it is
  proved, a *non*-garbling answer of `is_garbling_of` (`nothing`) cannot be certified; only
  the positive answer has a checked witness (`Certificate.RawGarbling.accepts_sound`).
* **T12b**: an action is E-admissible iff no mixed action strictly dominates it at every
  extreme point. Until it is proved, an action that `decide(problem, EAdmissible())` rejects
  cannot be certified; only accepted actions have a checked witness
  (`Certificate.RawEAdmissible.accepts_sound`).
* **T13b** (commitment restores monotonicity under Γ-maximin) is not attempted; its
  statement has not been checked against a source.

## Consequences for the package

Each line names a proved result and the Julia function or check it motivates.

* `minimaxRegret_violates_iia` (T8): the minimax-regret choice depends on the menu of
  actions, including actions that are never chosen. Done in Julia: the rule's result carries
  the note (`details.note`) and `DecisionReport` says so; `feasible_actions` filters change
  that choice.
* `evsi_eq_zero_iff` and `evsi_eq_zero_iff_of_pos` (T14): an experiment has zero EVSI exactly
  when one action is optimal after every signal of positive probability. This is a cheap test
  ("this information cannot change the decision") that needs one `argmax` per signal and no
  subtraction. Done in Julia: `information_can_change_decision(problem, experiment)`, held to
  `evsi > 0` by a property test.
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
* `valueWithInformation_garble_le` (T6) holds for a single prior only. `robust_evsi` must
  not assume it, and its documentation should keep saying so.
* `bayesAction_rescale` and `isMinimaxRegret_rescale`: `first_choice` is invariant under a
  change of payoff units for both rules, so reports need not record the units to be
  reproducible.
* `evppi_mono_of_factorsThrough`: `evppi` is monotone only between partitions one of which
  refines the other; two unrelated partitions are not ordered and the report should not rank
  them as if they were.
-/
