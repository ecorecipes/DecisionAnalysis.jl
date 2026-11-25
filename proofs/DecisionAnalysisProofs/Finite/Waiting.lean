import DecisionAnalysisProofs.Finite.ValueOfInformation
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!
# DecisionAnalysisProofs.Finite.Waiting

The value of waiting to learn (`value_of_waiting`, `src/design.jl`). Acting now is worth
`actNowValue u p`. Waiting means that the experiment `e` is observed and an action is then
taken late, with the payoffs `u'` of late action on the same outcome horizon. Nothing is
assumed about how `u'` relates to `u` unless a theorem says so.

## What each result backs or shapes

* **`valueOfWaiting_eq`**: the value of waiting is the value of the information *for the late
  decision* minus the cost of the delay itself, `evsi u' p e − delayCost u u' p`. Backs the
  fields `evsi_delayed` and `delay_cost` of the result and the test that they add up.
* **`valueOfWaiting_pos_iff`**: waiting is better than acting now exactly when that information
  is worth more than the delay costs. Backs the field `wait`.
* **`valueOfWaiting_constant`**: waiting and learning nothing is worth minus the cost of the
  delay.
* **`valueWithInformation_mono`** and **`valueOfWaiting_le_evsi`**: when no action is better
  late than now in any state, the value of waiting is at most the EVSI computed on the payoffs
  of acting now. Shapes the package: that "timeless" EVSI is reported beside the value of
  waiting as an upper bound, not as the value of the study, and the docstring says when the
  bound holds.
* **`valueOfWaiting_neg_example`**: an instance in which the experiment has positive EVSI on
  the payoffs of acting now and waiting for it is strictly worse than acting now.

Not claimed: anything about how the late payoffs are obtained (they come from the user's
model), or about more than one period of waiting.
-/

set_option autoImplicit false
-- One context is declared for the file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The value of waiting: observe `e`, then act late with payoffs `u'`, against acting now
with payoffs `u`. -/
def valueOfWaiting (u u' : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) : K :=
  valueWithInformation u' p e - actNowValue u p

/-- The cost of the delay alone: the best now against the best late, on the prior. -/
def delayCost (u u' : A → Θ → K) (p : Θ → K) : K := actNowValue u p - actNowValue u' p

theorem valueOfWaiting_eq (u u' : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) :
    valueOfWaiting u u' p e = evsi u' p e - delayCost u u' p := by
  simp only [valueOfWaiting, evsi, delayCost]
  ring

theorem valueOfWaiting_pos_iff (u u' : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) :
    0 < valueOfWaiting u u' p e ↔ delayCost u u' p < evsi u' p e := by
  rw [valueOfWaiting_eq, sub_pos]

theorem valueOfWaiting_constant (u u' : A → Θ → K) (p : Θ → K) :
    valueOfWaiting u u' p constantExperiment = -delayCost u u' p := by
  simp only [valueOfWaiting, delayCost, valueWithInformation_constant]
  ring

/-- The informed value is monotone in the payoffs. -/
theorem valueWithInformation_mono {u u' : A → Θ → K} {p : Θ → K} {e : Θ → Z → K}
    (hp : NonnegPrior p) (he : NonnegKernel e) (h : ∀ a θ, u' a θ ≤ u a θ) :
    valueWithInformation u' p e ≤ valueWithInformation u p e := by
  refine Finset.sum_le_sum fun z _ => maxOver_mono fun a => ?_
  exact Finset.sum_le_sum fun θ _ =>
    mul_le_mul_of_nonneg_left (h a θ) (mul_nonneg (hp θ) (he θ z))

/-- When no action is better late than now, the EVSI on the payoffs of acting now bounds the
value of waiting from above. -/
theorem valueOfWaiting_le_evsi {u u' : A → Θ → K} {p : Θ → K} {e : Θ → Z → K}
    (hp : NonnegPrior p) (he : NonnegKernel e) (h : ∀ a θ, u' a θ ≤ u a θ) :
    valueOfWaiting u u' p e ≤ evsi u p e := by
  simp only [valueOfWaiting, evsi]
  exact sub_le_sub_right (valueWithInformation_mono hp he h) _

/-- Two actions, two equally likely states, the state learned exactly. Now: action 0 pays
`(4, 0)` and action 1 pays `(0, 4)`, so the EVSI is 2. Late, every payoff is 3 lower. -/
theorem valueOfWaiting_neg_example :
    let u : Fin 2 → Fin 2 → ℚ := fun a θ => if a = θ then 4 else 0
    let u' : Fin 2 → Fin 2 → ℚ := fun a θ => if a = θ then 1 else -3
    let p : Fin 2 → ℚ := fun _ => 1 / 2
    let e : Fin 2 → Fin 2 → ℚ := fun θ z => if z = θ then 1 else 0
    evsi u p e = 2 ∧ valueOfWaiting u u' p e = -1 := by
  simp only [evsi, valueOfWaiting, valueWithInformation, actNowValue, signalValue,
    expectedUtility, maxOver_fin_two, Fin.sum_univ_two]
  constructor <;> norm_num

end DecisionAnalysisProofs
