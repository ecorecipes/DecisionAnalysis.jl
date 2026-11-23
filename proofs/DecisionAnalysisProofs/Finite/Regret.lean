import DecisionAnalysisProofs.Finite.Problem

/-!
# DecisionAnalysisProofs.Finite.Regret

Expected regret against expected utility (`src/regret.jl`, `src/voi.jl`).

* **T2**, **`minOver_expectedRegret_eq_evpi`**: the minimum expected regret equals EVPI;
  **`IsBayes.expectedRegret_eq_evpi`**: it is the expected regret of any Bayes action.
* **T3**, **`isBayes_iff_expectedRegret_le`** and **`setOf_isBayes_eq`**: the minimisers of
  expected regret are exactly the maximisers of expected utility.

Neither needs the prior to be nonnegative or normalised: both follow from
`expectedRegret u p a = perfectValue u p - expectedUtility u p a`.

Not claimed: any relation between *maximum* regret and expected utility.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

variable {A Θ : Type} [Fintype A] [Nonempty A] [Fintype Θ]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- **T2.** The minimum over actions of the expected regret is the EVPI
(`minimum(expected_regret(problem)) == evpi(problem)`). -/
theorem minOver_expectedRegret_eq_evpi (u : A → Θ → K) (p : Θ → K) :
    minOver (expectedRegret u p) = evpi u p := by
  have h : expectedRegret u p = fun a => perfectValue u p - expectedUtility u p a :=
    funext (expectedRegret_eq u p)
  rw [h, minOver_const_sub]
  rfl

/-- The expected regret of a Bayes action is the EVPI. -/
theorem IsBayes.expectedRegret_eq_evpi {u : A → Θ → K} {p : Θ → K} {a : A}
    (h : IsBayes u p a) : expectedRegret u p a = evpi u p := by
  rw [expectedRegret_eq, (isBayes_iff u p a).mp h]
  rfl

/-- **T3.** An action maximises expected utility iff it minimises expected regret. -/
theorem isBayes_iff_expectedRegret_le (u : A → Θ → K) (p : Θ → K) (a : A) :
    IsBayes u p a ↔ ∀ a', expectedRegret u p a ≤ expectedRegret u p a' := by
  simp only [expectedRegret_eq, IsBayes]
  constructor
  · intro h a'
    have := h a'
    linarith
  · intro h a'
    have := h a'
    linarith

/-- **T3**, as an equality of sets: argmax of expected utility = argmin of expected regret. -/
theorem setOf_isBayes_eq (u : A → Θ → K) (p : Θ → K) :
    {a | IsBayes u p a} = {a | ∀ a', expectedRegret u p a ≤ expectedRegret u p a'} :=
  Set.ext fun a => isBayes_iff_expectedRegret_le u p a

end DecisionAnalysisProofs
