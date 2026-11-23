import DecisionAnalysisProofs.Finite.Affine
import DecisionAnalysisProofs.Finite.Credal

/-!
# DecisionAnalysisProofs.Finite.Selection

Tie handling in rule selection (`src/rules.jl`: `decide(problem, rule; atol=0, rtol=0)` and
`first_choice`). With zero tolerances `decide` lists **every** action whose score attains the
optimum, in table order (`DecisionResult.chosen`), and `first_choice` returns the first of
them; ties are never broken at random. `maximisers`/`minimisers` are the chosen sets and
`firstArgmax`/`firstArgmin` the first choice, on a linearly ordered finite index type
(`Fin n` for the checker).

* **`mem_maximisers`**, **`mem_minimisers`**: the chosen set is exactly the set of optimal
  indices. **`firstArgmax_mem_maximisers`**: the first choice is one of them.
* **`firstArgmax_spec`**, **`le_firstArgmax_value`**: the first choice attains the maximum.
* **`lt_of_lt_firstArgmax`**: every earlier index is strictly worse (first among ties).
* **`eq_firstArgmax`**: those two properties determine the index, so the selection is
  deterministic: any implementation with them returns the same action.
* **`firstArgmax_congr_of_order`**: the selection depends only on the ranking of the scores,
  hence **`bayesAction_rescale`**: the first Bayes action is unchanged by a positive affine
  rescaling of the payoff table (stability of the reported decision under a change of units).
* **`isBayes_bayesAction`**, **`isMinimaxRegret_minimaxRegretAction`**,
  **`isGammaMaximin_gammaMaximinAction`**: the first choices satisfy the rule's
  specification.

Not claimed: stability under perturbation of the scores (ties can break either way under
arbitrarily small changes), and nothing about `decide` with nonzero `atol` or `rtol`.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

section Selector

variable {ι K : Type} [Fintype ι] [Nonempty ι] [LinearOrder ι] [LinearOrder K]

/-- The indices at which `f` attains its maximum (`DecisionResult.chosen`, sense `:max`). -/
def maximisers (f : ι → K) : Finset ι := Finset.univ.filter fun i => f i = maxOver f

/-- The indices at which `f` attains its minimum (`DecisionResult.chosen`, sense `:min`). -/
def minimisers (f : ι → K) : Finset ι := Finset.univ.filter fun i => f i = minOver f

theorem maximisers_nonempty (f : ι → K) : (maximisers f).Nonempty := by
  obtain ⟨i, hi⟩ := exists_eq_maxOver f
  exact ⟨i, by simp [maximisers, hi]⟩

theorem minimisers_nonempty (f : ι → K) : (minimisers f).Nonempty := by
  obtain ⟨i, hi⟩ := exists_eq_minOver f
  exact ⟨i, by simp [minimisers, hi]⟩

/-- `first_choice` of a maximising rule: the first index attaining the maximum. -/
def firstArgmax (f : ι → K) : ι := (maximisers f).min' (maximisers_nonempty f)

/-- `first_choice` of a minimising rule: the first index attaining the minimum. -/
def firstArgmin (f : ι → K) : ι := (minimisers f).min' (minimisers_nonempty f)

theorem mem_maximisers (f : ι → K) (i : ι) : i ∈ maximisers f ↔ ∀ j, f j ≤ f i := by
  simp only [maximisers, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · intro h j
    rw [h]
    exact le_maxOver f j
  · intro h
    exact (maxOver_eq_of i rfl h).symm

theorem mem_minimisers (f : ι → K) (i : ι) : i ∈ minimisers f ↔ ∀ j, f i ≤ f j := by
  simp only [minimisers, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · intro h j
    rw [h]
    exact minOver_le f j
  · intro h
    exact (minOver_eq_of i rfl h).symm

theorem firstArgmax_mem_maximisers (f : ι → K) : firstArgmax f ∈ maximisers f :=
  Finset.min'_mem (maximisers f) (maximisers_nonempty f)

theorem firstArgmin_mem_minimisers (f : ι → K) : firstArgmin f ∈ minimisers f :=
  Finset.min'_mem (minimisers f) (minimisers_nonempty f)

theorem firstArgmax_spec (f : ι → K) : f (firstArgmax f) = maxOver f :=
  (Finset.mem_filter.mp (Finset.min'_mem (maximisers f) (maximisers_nonempty f))).2

theorem firstArgmin_spec (f : ι → K) : f (firstArgmin f) = minOver f :=
  (Finset.mem_filter.mp (Finset.min'_mem (minimisers f) (minimisers_nonempty f))).2

theorem le_firstArgmax_value (f : ι → K) (i : ι) : f i ≤ f (firstArgmax f) := by
  exact (le_maxOver f i).trans (firstArgmax_spec f).ge

theorem firstArgmin_value_le (f : ι → K) (i : ι) : f (firstArgmin f) ≤ f i := by
  exact (firstArgmin_spec f).le.trans (minOver_le f i)

theorem firstArgmax_le {f : ι → K} {i : ι} (hi : ∀ j, f j ≤ f i) : firstArgmax f ≤ i :=
  Finset.min'_le _ _ (Finset.mem_filter.mpr ⟨Finset.mem_univ i, (maxOver_eq_of i rfl hi).symm⟩)

theorem firstArgmin_le {f : ι → K} {i : ι} (hi : ∀ j, f i ≤ f j) : firstArgmin f ≤ i :=
  Finset.min'_le _ _ (Finset.mem_filter.mpr ⟨Finset.mem_univ i, (minOver_eq_of i rfl hi).symm⟩)

/-- First among ties: every index before the selected one is strictly worse. -/
theorem lt_of_lt_firstArgmax {f : ι → K} {j : ι} (hj : j < firstArgmax f) :
    f j < f (firstArgmax f) := by
  by_contra h
  have h' : ∀ k, f k ≤ f j := fun k => (le_firstArgmax_value f k).trans (not_lt.mp h)
  exact absurd (firstArgmax_le h') (not_le.mpr hj)

theorem lt_of_lt_firstArgmin {f : ι → K} {j : ι} (hj : j < firstArgmin f) :
    f (firstArgmin f) < f j := by
  by_contra h
  have h' : ∀ k, f j ≤ f k := fun k => (not_lt.mp h).trans (firstArgmin_value_le f k)
  exact absurd (firstArgmin_le h') (not_le.mpr hj)

/-- Determinism: an index that attains the maximum and beats every earlier index strictly is
the selected one. -/
theorem eq_firstArgmax {f : ι → K} {i : ι} (hmax : ∀ j, f j ≤ f i)
    (hfirst : ∀ j, j < i → f j < f i) : firstArgmax f = i := by
  apply le_antisymm (firstArgmax_le hmax)
  by_contra h
  have h1 := hfirst _ (not_le.mp h)
  have h2 := le_firstArgmax_value f i
  exact lt_irrefl _ (lt_of_lt_of_le h1 h2)

theorem eq_firstArgmin {f : ι → K} {i : ι} (hmin : ∀ j, f i ≤ f j)
    (hfirst : ∀ j, j < i → f i < f j) : firstArgmin f = i := by
  apply le_antisymm (firstArgmin_le hmin)
  by_contra h
  have h1 := hfirst _ (not_le.mp h)
  have h2 := firstArgmin_value_le f i
  exact lt_irrefl _ (lt_of_lt_of_le h1 h2)

/-- The selection depends only on the ranking of the scores. -/
theorem firstArgmax_congr_of_order {K' : Type} [LinearOrder K'] {f : ι → K} {g : ι → K'}
    (h : ∀ i j, f i ≤ f j ↔ g i ≤ g j) : firstArgmax g = firstArgmax f := by
  apply eq_firstArgmax
  · intro j
    exact (h _ _).mp (le_firstArgmax_value f j)
  · intro j hj
    have h1 := lt_of_lt_firstArgmax hj
    exact not_le.mp fun hg => absurd ((h _ _).mpr hg) (not_le.mpr h1)

end Selector

section Rules

variable {A Θ I : Type} [Fintype A] [Nonempty A] [LinearOrder A] [Fintype Θ] [Fintype I]
  [Nonempty I] {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- `first_choice(decide(problem, MaxExpectedUtility()))`: the first action of maximal
expected utility. -/
def bayesAction (u : A → Θ → K) (p : Θ → K) : A := firstArgmax (expectedUtility u p)

/-- `first_choice(decide(problem, MinimaxRegret()))`: the first action of minimal maximum
regret. -/
def minimaxRegretAction [Nonempty Θ] (u : A → Θ → K) : A := firstArgmin (maxRegret u)

/-- `first_choice(decide(problem, GammaMaximin()))`: the first action of maximal lower
expected utility. -/
def gammaMaximinAction (u : A → Θ → K) (P : I → Θ → K) : A :=
  firstArgmax (lowerExpectedUtility u P)

theorem isBayes_bayesAction (u : A → Θ → K) (p : Θ → K) : IsBayes u p (bayesAction u p) :=
  le_firstArgmax_value (expectedUtility u p)

theorem isMinimaxRegret_minimaxRegretAction [Nonempty Θ] (u : A → Θ → K) :
    IsMinimaxRegret u (minimaxRegretAction u) :=
  firstArgmin_value_le (maxRegret u)

theorem isGammaMaximin_gammaMaximinAction (u : A → Θ → K) (P : I → Θ → K) :
    IsGammaMaximin u P (gammaMaximinAction u P) :=
  le_firstArgmax_value (lowerExpectedUtility u P)

/-- The selected Bayes action is unchanged by a positive affine rescaling of the payoffs. -/
theorem bayesAction_rescale {s : K} (hs : 0 < s) (b : Θ → K) (u : A → Θ → K) (p : Θ → K) :
    bayesAction (rescale s b u) p = bayesAction u p := by
  apply firstArgmax_congr_of_order
  intro i j
  rw [expectedUtility_rescale, expectedUtility_rescale]
  constructor
  · intro h
    have := mul_le_mul_of_nonneg_left h hs.le
    linarith
  · intro h
    exact le_of_mul_le_mul_left (by linarith) hs

end Rules

end DecisionAnalysisProofs
