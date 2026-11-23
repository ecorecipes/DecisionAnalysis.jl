import DecisionAnalysisProofs.Certificate.Witness

/-!
# DecisionAnalysisProofs.Certificate.Refutation

Checkers for certified **"no"** answers. `decide(problem, EAdmissible())` and
`garbling_result` (`src/rules.jl`, `src/experiments.jl`) mark a negative answer exact only when
it comes from an exact route; from a floating-point optimiser it is flagged numerical. With
the witnesses below, a negative answer is checked in exact arithmetic whatever produced it,
exactly as positive answers already are. The schemas are specified in
`docs/src/certificates.md`; the package writes them with `export_not_eadmissible_witness` and
`export_not_garbling_witness`.

## `"not-e-admissible-witness/1"`

Given the payoff table, the extreme points of the credal set, an action and a **mixed action**
(a probability vector over actions), `RawNotEAdmissible.Accepts` checks the shapes, that the
extreme points are probability vectors, that the mixed action is nonnegative and sums to one,
and that its expected utility is strictly larger than the action's at **every** extreme point.

* **`MixedDominanceWitnessOk.not_isEAdmissible`**, **`RawNotEAdmissible.accepts_sound`**: an
  accepted witness proves `¬ IsEAdmissible` for the convex hull of the extreme points.
* Complete by `not_isEAdmissible_iff_isMixedDominated` (`Finite/Alternative.lean`): every
  action that is not E-admissible has such a witness. The mixed action is the dual solution
  of `eadmissibility_model` (the multipliers of the rows `optimal[b]`).

## `"not-garbling-witness/1"`

Given two experiments on the same states, a payoff table and a prior,
`RawNotGarbling.Accepts` checks the shapes, that both likelihoods are column-stochastic, that
the prior is a probability vector, and that the value of deciding after `garbled` is
**strictly larger** than after `experiment` (both recomputed with `valueWithInformation`).

* **`RawNotGarbling.accepts_sound`**: an accepted witness proves `¬ IsGarblingOf e′ e`, by
  T6 (`valueWithInformation_garble_le`): a garbling is never worth more. No duality is used.
* Complete by `not_isGarblingOf_iff_exists_separating` (`Finite/Alternative.lean`): whenever
  `e′` is not a garbling of `e` there is a separating payoff table, and it can be taken with
  one action per signal of `e′` and the uniform prior.

Edge cases. A document with no extreme points is accepted by the first checker (no action is
E-admissible for an empty credal set; `IsEAdmissible` then has no weights summing to one);
Julia's `CredalSet` cannot be empty, so the exporter never writes one. A payoff table with no
actions is rejected by both checkers.

Not claimed: correctness of the JSON parser in `CheckCertificate.lean`, which is trusted, and
nothing about how Julia obtains the witnesses.
-/

set_option autoImplicit false

namespace DecisionAnalysisProofs.Certificate

open DecisionAnalysisProofs

/-! ### Not E-admissible -/

/-- `m` is a probability vector over actions whose expected utility is strictly larger than
that of `a` under every member of the family `P`. -/
def MixedDominanceWitnessOk {n k s : ℕ} (u : Fin n → Fin s → ℚ) (P : Fin k → Fin s → ℚ)
    (m : Fin n → ℚ) (a : Fin n) : Prop :=
  (∀ b, 0 ≤ m b) ∧ ∑ b, m b = 1 ∧
    ∀ i, expectedUtility u (P i) a < ∑ b, m b * expectedUtility u (P i) b

instance {n k s : ℕ} (u : Fin n → Fin s → ℚ) (P : Fin k → Fin s → ℚ) (m : Fin n → ℚ)
    (a : Fin n) : Decidable (MixedDominanceWitnessOk u P m a) := by
  unfold MixedDominanceWitnessOk; infer_instance

/-- **Soundness of the mixed-dominance witness.** -/
theorem MixedDominanceWitnessOk.not_isEAdmissible {n k s : ℕ} {u : Fin n → Fin s → ℚ}
    {P : Fin k → Fin s → ℚ} {m : Fin n → ℚ} {a : Fin n}
    (h : MixedDominanceWitnessOk u P m a) : ¬ IsEAdmissible u P a :=
  IsMixedDominated.not_isEAdmissible ⟨m, h.1, h.2.1, h.2.2⟩

/-- A decoded `"not-e-admissible-witness/1"` document. -/
structure RawNotEAdmissible where
  /-- The number of states (columns of the payoff table and of the priors). -/
  nStates : ℕ
  /-- The payoff table: rows are actions, columns are states. -/
  payoff : List (List ℚ)
  /-- The extreme points of the credal set: one prior per row. -/
  priors : List (List ℚ)
  /-- The dominating mixed action: one probability per action. -/
  mixedAction : List ℚ
  /-- The zero-based index of the action claimed not to be E-admissible. -/
  action : ℕ

/-- The payoff table. -/
def RawNotEAdmissible.u (r : RawNotEAdmissible) :
    Fin r.mixedAction.length → Fin r.nStates → ℚ :=
  table r.payoff r.mixedAction.length r.nStates

/-- The extreme points. -/
def RawNotEAdmissible.P (r : RawNotEAdmissible) : Fin r.priors.length → Fin r.nStates → ℚ :=
  table r.priors r.priors.length r.nStates

/-- What the not-E-admissible checker decides. -/
def RawNotEAdmissible.Accepts (r : RawNotEAdmissible) : Prop :=
  rect r.payoff r.mixedAction.length r.nStates = true ∧
  rect r.priors r.priors.length r.nStates = true ∧
  (∀ i, NonnegPrior (r.P i) ∧ NormalisedPrior (r.P i)) ∧
  ∃ h : r.action < r.mixedAction.length,
    MixedDominanceWitnessOk r.u r.P (vec r.mixedAction r.mixedAction.length) ⟨r.action, h⟩

instance (r : RawNotEAdmissible) : Decidable r.Accepts := by
  unfold RawNotEAdmissible.Accepts; infer_instance

/-- The not-E-admissible checker run by `lake exe check_certificate`. -/
def checkNotEAdmissible (r : RawNotEAdmissible) : Bool := decide r.Accepts

/-- **Soundness of the not-E-admissible checker.** If it accepts, the action is not
E-admissible for the credal set generated by the stated extreme points: no prior in their
convex hull makes it a Bayes action. -/
theorem RawNotEAdmissible.accepts_sound {r : RawNotEAdmissible}
    (h : checkNotEAdmissible r = true) :
    ∃ ha : r.action < r.mixedAction.length, ¬ IsEAdmissible r.u r.P ⟨r.action, ha⟩ := by
  have h' : r.Accepts := of_decide_eq_true h
  obtain ⟨-, -, -, ha, hw⟩ := h'
  exact ⟨ha, hw.not_isEAdmissible⟩

/-! ### Not a garbling -/

/-- A decoded `"not-garbling-witness/1"` document. -/
structure RawNotGarbling where
  /-- The number of states. -/
  nStates : ℕ
  /-- The likelihood of `e`: rows are signals, columns are states. -/
  experiment : List (List ℚ)
  /-- The likelihood of `e′`, claimed not to be a garbling of `e`. -/
  garbled : List (List ℚ)
  /-- The separating payoff table: rows are actions, columns are states. -/
  payoff : List (List ℚ)
  /-- The prior of the separating problem. -/
  prior : List ℚ

/-- The experiment `e`. -/
def RawNotGarbling.e (r : RawNotGarbling) : Fin r.nStates → Fin r.experiment.length → ℚ :=
  kernelOf r.experiment r.nStates r.experiment.length

/-- The experiment `e′`. -/
def RawNotGarbling.e' (r : RawNotGarbling) : Fin r.nStates → Fin r.garbled.length → ℚ :=
  kernelOf r.garbled r.nStates r.garbled.length

/-- The separating payoff table; the number of actions is written `tail.length + 1` so that
the action type is visibly nonempty. -/
def RawNotGarbling.u (r : RawNotGarbling) :
    Fin (r.payoff.tail.length + 1) → Fin r.nStates → ℚ :=
  table r.payoff (r.payoff.tail.length + 1) r.nStates

/-- The prior. -/
def RawNotGarbling.p (r : RawNotGarbling) : Fin r.nStates → ℚ := vec r.prior r.nStates

/-- What the not-a-garbling checker decides. -/
def RawNotGarbling.Accepts (r : RawNotGarbling) : Prop :=
  rect r.payoff (r.payoff.tail.length + 1) r.nStates = true ∧
  rect r.experiment r.experiment.length r.nStates = true ∧
  rect r.garbled r.garbled.length r.nStates = true ∧
  r.prior.length = r.nStates ∧
  StochasticKernel r.e ∧ StochasticKernel r.e' ∧
  NonnegPrior r.p ∧ NormalisedPrior r.p ∧
  valueWithInformation r.u r.p r.e < valueWithInformation r.u r.p r.e'

instance (r : RawNotGarbling) : Decidable r.Accepts := by
  unfold RawNotGarbling.Accepts; infer_instance

/-- The not-a-garbling checker run by `lake exe check_certificate`. -/
def checkNotGarbling (r : RawNotGarbling) : Bool := decide r.Accepts

/-- **Soundness of the not-a-garbling checker.** If it accepts, `e′` is not a garbling of
`e`: the stated decision problem, with a probability prior and stochastic experiments, is one
in which deciding after `e′` is worth strictly more. -/
theorem RawNotGarbling.accepts_sound {r : RawNotGarbling} (h : checkNotGarbling r = true) :
    ¬ IsGarblingOf r.e' r.e ∧ StochasticKernel r.e ∧ StochasticKernel r.e' ∧
      NonnegPrior r.p ∧ NormalisedPrior r.p ∧
      valueWithInformation r.u r.p r.e < valueWithInformation r.u r.p r.e' := by
  have h' : r.Accepts := of_decide_eq_true h
  obtain ⟨-, -, -, -, he, he', hp, hp1, hlt⟩ := h'
  exact ⟨fun hg => absurd (hg.valueWithInformation_le r.u r.p) (not_le.mpr hlt),
    he, he', hp, hp1, hlt⟩

end DecisionAnalysisProofs.Certificate
