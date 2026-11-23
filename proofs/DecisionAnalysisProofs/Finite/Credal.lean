import DecisionAnalysisProofs.Finite.Problem

/-!
# DecisionAnalysisProofs.Finite.Credal

Decision rules for a finite nonempty family of priors `P : I → Θ → K`, the extreme points of a
`CredalSet` (`src/beliefs.jl`, `src/rules.jl`: `GammaMaximin`, `EAdmissible`, `Maximal`,
`IntervalDominance`).

## Conventions

* `lowerExpectedUtility u P a` is the minimum over the members of the expected utility of `a`;
  `gammaMaximinValue` is its maximum over actions; `IsGammaMaximin` is attainment.
* `IsEAdmissible u P a` quantifies over the **convex hull** of the family: `a` is Bayes for
  some mixture `Σ_i w i · P i` with nonnegative weights summing to one. Checking only the
  members would miss actions that are optimal at interior priors only.
* `IsMaximal u P a`: no action has strictly larger expected utility under every member. By
  `expectedUtility_mixture` strict dominance on the members is strict dominance on the hull
  (`IsMaximal.not_dominated_on_hull`).
* `IsIntervalUndominated u P a`: no action's lower expected utility exceeds the upper expected
  utility of `a`.

## Results (T12)

* **`lowerExpectedUtility_le`**, **`IsGammaMaximin.gammaMaximinValue_le`**,
  **`gammaMaximinValue_le_actNowValue`**: the Γ-maximin value is at most each member's expected
  utility of the chosen action, and at most each member's act-now value.
* **`IsBayes.isEAdmissible`**: each member's Bayes action is E-admissible.
* **`IsEAdmissible.isMaximal`**: E-admissible ⊆ maximal.
* **`IsMaximal.isIntervalUndominated`**: maximal ⊆ interval-undominated.
* **`IsGammaMaximin.isMaximal`**: every Γ-maximin action is maximal.

Not claimed: the converse characterisation of E-admissibility by mixed-action dominance (T12b
in `Roadmap.lean`), and nothing about the linear programme Julia uses to decide it.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ I : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype I] [Nonempty I]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The lower expected utility of `a` over the family of priors (`LowerExpectation`). -/
def lowerExpectedUtility (u : A → Θ → K) (P : I → Θ → K) (a : A) : K :=
  minOver fun i => expectedUtility u (P i) a

/-- The upper expected utility of `a` over the family of priors (`UpperExpectation`). -/
def upperExpectedUtility (u : A → Θ → K) (P : I → Θ → K) (a : A) : K :=
  maxOver fun i => expectedUtility u (P i) a

/-- The Γ-maximin value: the best lower expected utility. -/
def gammaMaximinValue (u : A → Θ → K) (P : I → Θ → K) : K :=
  maxOver (lowerExpectedUtility u P)

/-- `a` is a Γ-maximin action (`GammaMaximin`). -/
def IsGammaMaximin (u : A → Θ → K) (P : I → Θ → K) (a : A) : Prop :=
  ∀ a', lowerExpectedUtility u P a' ≤ lowerExpectedUtility u P a

/-- The mixture `Σ_i w i · P i` of the family. -/
def mixture (w : I → K) (P : I → Θ → K) : Θ → K := fun θ => ∑ i, w i * P i θ

/-- `a` is E-admissible (`EAdmissible`): Bayes for some prior in the convex hull. -/
def IsEAdmissible (u : A → Θ → K) (P : I → Θ → K) (a : A) : Prop :=
  ∃ w : I → K, (∀ i, 0 ≤ w i) ∧ ∑ i, w i = 1 ∧ IsBayes u (mixture w P) a

/-- `a` is maximal (`Maximal`): no action is strictly better under every member. -/
def IsMaximal (u : A → Θ → K) (P : I → Θ → K) (a : A) : Prop :=
  ¬ ∃ a', ∀ i, expectedUtility u (P i) a < expectedUtility u (P i) a'

/-- `a` is not interval-dominated (`IntervalDominance`). -/
def IsIntervalUndominated (u : A → Θ → K) (P : I → Θ → K) (a : A) : Prop :=
  ¬ ∃ a', upperExpectedUtility u P a < lowerExpectedUtility u P a'

/-- **T12.** The lower expected utility is below each member's expected utility. -/
theorem lowerExpectedUtility_le (u : A → Θ → K) (P : I → Θ → K) (a : A) (i : I) :
    lowerExpectedUtility u P a ≤ expectedUtility u (P i) a :=
  minOver_le (fun i => expectedUtility u (P i) a) i

theorem le_upperExpectedUtility (u : A → Θ → K) (P : I → Θ → K) (a : A) (i : I) :
    expectedUtility u (P i) a ≤ upperExpectedUtility u P a :=
  le_maxOver (fun i => expectedUtility u (P i) a) i

theorem isGammaMaximin_iff (u : A → Θ → K) (P : I → Θ → K) (a : A) :
    IsGammaMaximin u P a ↔ lowerExpectedUtility u P a = gammaMaximinValue u P := by
  constructor
  · intro h
    exact (maxOver_eq_of a rfl h).symm
  · intro h a'
    rw [h]
    exact le_maxOver _ a'

/-- A Γ-maximin action exists. -/
theorem exists_isGammaMaximin (u : A → Θ → K) (P : I → Θ → K) : ∃ a, IsGammaMaximin u P a := by
  obtain ⟨a, ha⟩ := exists_eq_maxOver (lowerExpectedUtility u P)
  exact ⟨a, (isGammaMaximin_iff u P a).mpr ha.symm⟩

/-- **T12.** The Γ-maximin value is at most each member's expected utility of the chosen
action. -/
theorem IsGammaMaximin.gammaMaximinValue_le {u : A → Θ → K} {P : I → Θ → K} {a : A}
    (h : IsGammaMaximin u P a) (i : I) :
    gammaMaximinValue u P ≤ expectedUtility u (P i) a := by
  rw [← (isGammaMaximin_iff u P a).mp h]
  exact lowerExpectedUtility_le u P a i

/-- **T12.** The Γ-maximin value is at most each member's act-now value. -/
theorem gammaMaximinValue_le_actNowValue (u : A → Θ → K) (P : I → Θ → K) (i : I) :
    gammaMaximinValue u P ≤ actNowValue u (P i) :=
  maxOver_mono fun a => lowerExpectedUtility_le u P a i

omit [Nonempty A] [Nonempty I] in
/-- Expected utility is linear in the prior. -/
theorem expectedUtility_mixture (u : A → Θ → K) (w : I → K) (P : I → Θ → K) (a : A) :
    expectedUtility u (mixture w P) a = ∑ i, w i * expectedUtility u (P i) a := by
  simp only [expectedUtility, mixture, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun θ _ => by ring

/-- **T12.** A Bayes action of a member of the family is E-admissible. -/
theorem IsBayes.isEAdmissible {u : A → Θ → K} {P : I → Θ → K} {a : A} {i : I}
    (h : IsBayes u (P i) a) : IsEAdmissible u P a := by
  classical
  refine ⟨fun j => if j = i then 1 else 0, fun j => ?_, by simp, ?_⟩
  · by_cases hj : j = i <;> simp [hj]
  · have hm : mixture (fun j => if j = i then (1 : K) else 0) P = P i := by
      funext θ
      simp [mixture]
    rw [hm]
    exact h

omit [Nonempty A] [Nonempty I] in
/-- Strict dominance on every member is strict dominance on every mixture. -/
theorem expectedUtility_mixture_lt {u : A → Θ → K} {P : I → Θ → K} {a a' : A}
    (h : ∀ i, expectedUtility u (P i) a < expectedUtility u (P i) a') {w : I → K}
    (hw : ∀ i, 0 ≤ w i) (hw1 : ∑ i, w i = 1) :
    expectedUtility u (mixture w P) a < expectedUtility u (mixture w P) a' := by
  obtain ⟨i, hi⟩ : ∃ i, 0 < w i := by
    by_contra hne
    have : ∑ i, w i ≤ 0 := Finset.sum_nonpos fun i _ => not_lt.mp fun hi => hne ⟨i, hi⟩
    linarith
  rw [expectedUtility_mixture, expectedUtility_mixture]
  exact Finset.sum_lt_sum (fun j _ => mul_le_mul_of_nonneg_left (h j).le (hw j))
    ⟨i, Finset.mem_univ i, mul_lt_mul_of_pos_left (h i) hi⟩

omit [Nonempty A] [Nonempty I] in
/-- **T12.** Every E-admissible action is maximal. -/
theorem IsEAdmissible.isMaximal {u : A → Θ → K} {P : I → Θ → K} {a : A}
    (h : IsEAdmissible u P a) : IsMaximal u P a := by
  obtain ⟨w, hw, hw1, hB⟩ := h
  rintro ⟨a', hdom⟩
  have h1 := expectedUtility_mixture_lt hdom hw hw1
  have h2 := hB a'
  linarith

/-- Every Γ-maximin action is maximal. -/
theorem IsGammaMaximin.isMaximal {u : A → Θ → K} {P : I → Θ → K} {a : A}
    (h : IsGammaMaximin u P a) : IsMaximal u P a := by
  rintro ⟨a', hdom⟩
  obtain ⟨i, hi⟩ := exists_eq_minOver (fun i => expectedUtility u (P i) a')
  have h1 := h a'
  have h2 := lowerExpectedUtility_le u P a i
  have h3 := hdom i
  have h4 : lowerExpectedUtility u P a' = expectedUtility u (P i) a' := hi
  linarith

/-- A maximal action is not strictly dominated on any prior of the convex hull either. -/
theorem IsMaximal.not_dominated_on_hull {u : A → Θ → K} {P : I → Θ → K} {a : A}
    (h : IsMaximal u P a) :
    ¬ ∃ a', ∀ w : I → K, (∀ i, 0 ≤ w i) → ∑ i, w i = 1 →
      expectedUtility u (mixture w P) a < expectedUtility u (mixture w P) a' := by
  classical
  rintro ⟨a', hdom⟩
  refine h ⟨a', fun i => ?_⟩
  have hm : mixture (fun j => if j = i then (1 : K) else 0) P = P i := by
    funext θ
    simp [mixture]
  have h1 := hdom (fun j => if j = i then 1 else 0)
    (fun j => by by_cases hj : j = i <;> simp [hj]) (by simp)
  rwa [hm] at h1

/-- **T12.** Every maximal action is interval-undominated. -/
theorem IsMaximal.isIntervalUndominated {u : A → Θ → K} {P : I → Θ → K} {a : A}
    (h : IsMaximal u P a) : IsIntervalUndominated u P a := by
  rintro ⟨a', hdom⟩
  refine h ⟨a', fun i => ?_⟩
  have h1 := le_upperExpectedUtility u P a i
  have h2 := lowerExpectedUtility_le u P a' i
  linarith

end DecisionAnalysisProofs
