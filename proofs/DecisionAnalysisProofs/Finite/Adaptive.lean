import DecisionAnalysisProofs.Finite.ValueOfInformation

/-!
# DecisionAnalysisProofs.Finite.Adaptive

Sequential (adaptive) management over a finite horizon (`adaptive_management`,
`src/adaptive.jl`). The state `θ` is fixed and unknown. In each period an action `a` is taken,
pays `u a θ`, and reveals a signal `z` with probability `e a θ z`: what is learned depends on
what is done. Later periods are discounted by `δ`.

Beliefs are carried **unnormalised**, as the package does everywhere ("the joint form"): after
action `a` and signal `z` the weights `w θ` become `w θ * e a θ z`, with no division, so a
signal of probability zero contributes zero and needs no special case.

* `adaptiveValue u e δ n w`: the value of the best policy over `n` periods (the recursion that
  `adaptive_management` evaluates; "active" adaptive management).
* `policyValue u e δ π n w`: the value of a policy `π`, which chooses an action from the number
  of periods left and the current weights.
* `horizonWeight δ n = 1 + δ + … + δ^(n-1)`.

## What each result backs or shapes

* **`policyValue_le_adaptiveValue`** and **`exists_policyValue_eq_adaptiveValue`**: the
  recursion is the value of the best policy. In particular the passive policy (take the Bayes
  action of the current belief, learn from what happens, never act in order to learn) is worth
  at most the active value. Backs `value_of_probing ≥ 0`.
* **`horizonWeight_mul_actNowValue_le_policyValue`**: a policy that takes a Bayes action at
  every belief is worth at least the best fixed action repeated with no updating. Backs
  `passive ≥ no_learning`.
* **`horizonWeight_mul_actNowValue_le_adaptiveValue`**: so is the active value. Backs
  `value_of_learning ≥ 0`.
* **`adaptiveValue_le_horizonWeight_mul_perfectValue`**: no policy does better than knowing
  the state from the start. Backs `active ≤ perfect`, that is, the value of learning is at
  most the EVPI of one period times the discounted number of periods.

Not claimed: rewards or experiments that change from period to period, a state that changes
over time (the recursion is for a fixed unknown state), or anything about how the recursion is
memoised in the package.
-/

set_option autoImplicit false
-- One context is declared for the file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The weights after action `a` and signal `z`, unnormalised. -/
def updateWeights (e : A → Θ → Z → K) (w : Θ → K) (a : A) (z : Z) : Θ → K :=
  fun θ => w θ * e a θ z

/-- The value of the best policy over `n` periods from the weights `w`. -/
def adaptiveValue (u : A → Θ → K) (e : A → Θ → Z → K) (δ : K) : ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => maxOver fun a =>
      expectedUtility u w a + δ * ∑ z, adaptiveValue u e δ n (updateWeights e w a z)

/-- The value of the policy `π` over `n` periods from the weights `w`. -/
def policyValue (u : A → Θ → K) (e : A → Θ → Z → K) (δ : K) (π : ℕ → (Θ → K) → A) :
    ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => expectedUtility u w (π n w) +
      δ * ∑ z, policyValue u e δ π n (updateWeights e w (π n w) z)

/-- `1 + δ + … + δ^(n-1)`. -/
def horizonWeight (δ : K) : ℕ → K
  | 0 => 0
  | n + 1 => 1 + δ * horizonWeight δ n

theorem horizonWeight_nonneg {δ : K} (hδ : 0 ≤ δ) (n : ℕ) : 0 ≤ horizonWeight δ n := by
  induction n with
  | zero => exact le_refl _
  | succ n ih => exact add_nonneg zero_le_one (mul_nonneg hδ ih)

/-- No policy is worth more than the recursion. -/
theorem policyValue_le_adaptiveValue (u : A → Θ → K) (e : A → Θ → Z → K) {δ : K} (hδ : 0 ≤ δ)
    (π : ℕ → (Θ → K) → A) (n : ℕ) (w : Θ → K) :
    policyValue u e δ π n w ≤ adaptiveValue u e δ n w := by
  induction n generalizing w with
  | zero => exact le_refl _
  | succ n ih =>
    refine le_trans ?_ (le_maxOver _ (π n w))
    have h := mul_le_mul_of_nonneg_left
      (Finset.sum_le_sum fun z (_ : z ∈ Finset.univ) =>
        ih (updateWeights e w (π n w) z)) hδ
    show expectedUtility u w (π n w) +
        δ * ∑ z, policyValue u e δ π n (updateWeights e w (π n w) z) ≤
      expectedUtility u w (π n w) +
        δ * ∑ z, adaptiveValue u e δ n (updateWeights e w (π n w) z)
    linarith

/-- The recursion is attained by a policy. -/
theorem exists_policyValue_eq_adaptiveValue (u : A → Θ → K) (e : A → Θ → Z → K) (δ : K) :
    ∃ π : ℕ → (Θ → K) → A, ∀ n w, policyValue u e δ π n w = adaptiveValue u e δ n w := by
  classical
  let f : ℕ → (Θ → K) → A → K := fun n w a =>
    expectedUtility u w a + δ * ∑ z, adaptiveValue u e δ n (updateWeights e w a z)
  let π : ℕ → (Θ → K) → A := fun n w => Classical.choose (exists_eq_maxOver (f n w))
  have hπ : ∀ n w, maxOver (f n w) = f n w (π n w) := fun n w =>
    Classical.choose_spec (exists_eq_maxOver (f n w))
  refine ⟨π, fun n => ?_⟩
  induction n with
  | zero => intro w; rfl
  | succ n ih =>
    intro w
    have step : policyValue u e δ π (n + 1) w = f n w (π n w) := by
      show expectedUtility u w (π n w) +
          δ * ∑ z, policyValue u e δ π n (updateWeights e w (π n w) z) =
        expectedUtility u w (π n w) +
          δ * ∑ z, adaptiveValue u e δ n (updateWeights e w (π n w) z)
      rw [Finset.sum_congr rfl fun z _ => ih (updateWeights e w (π n w) z)]
    exact step.trans (hπ n w).symm

/-- Acting after a signal is, summed over signals, at least acting now. -/
theorem actNowValue_le_sum_updateWeights (u : A → Θ → K) {e : A → Θ → Z → K}
    (he : ∀ a, NormalisedKernel (e a)) (w : Θ → K) (a : A) :
    actNowValue u w ≤ ∑ z, actNowValue u (updateWeights e w a z) := by
  have h := evsi_nonneg u w (he a)
  unfold evsi at h
  have h' : valueWithInformation u w (e a) = ∑ z, actNowValue u (updateWeights e w a z) := rfl
  linarith

/-- A policy that takes a Bayes action at every belief is worth at least the best fixed action
with no updating. -/
theorem horizonWeight_mul_actNowValue_le_policyValue (u : A → Θ → K) {e : A → Θ → Z → K}
    (he : ∀ a, NormalisedKernel (e a)) {δ : K} (hδ : 0 ≤ δ) {π : ℕ → (Θ → K) → A}
    (hπ : ∀ n w, IsBayes u w (π n w)) (n : ℕ) (w : Θ → K) :
    horizonWeight δ n * actNowValue u w ≤ policyValue u e δ π n w := by
  induction n generalizing w with
  | zero => simp [horizonWeight, policyValue]
  | succ n ih =>
    have hb : expectedUtility u w (π n w) = actNowValue u w := (isBayes_iff u w _).mp (hπ n w)
    have h1 : horizonWeight δ n * actNowValue u w ≤
        ∑ z, policyValue u e δ π n (updateWeights e w (π n w) z) := by
      calc horizonWeight δ n * actNowValue u w
          ≤ horizonWeight δ n * ∑ z, actNowValue u (updateWeights e w (π n w) z) :=
            mul_le_mul_of_nonneg_left (actNowValue_le_sum_updateWeights u he w _)
              (horizonWeight_nonneg hδ n)
        _ = ∑ z, horizonWeight δ n * actNowValue u (updateWeights e w (π n w) z) :=
            Finset.mul_sum _ _ _
        _ ≤ _ := Finset.sum_le_sum fun z _ => ih _
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    simp only [horizonWeight, policyValue, hb]
    linarith

/-- The active value is at least the best fixed action with no updating. -/
theorem horizonWeight_mul_actNowValue_le_adaptiveValue (u : A → Θ → K) {e : A → Θ → Z → K}
    (he : ∀ a, NormalisedKernel (e a)) {δ : K} (hδ : 0 ≤ δ) (n : ℕ) (w : Θ → K) :
    horizonWeight δ n * actNowValue u w ≤ adaptiveValue u e δ n w := by
  classical
  let π : ℕ → (Θ → K) → A := fun _ w => Classical.choose (exists_isBayes u w)
  have hπ : ∀ n w, IsBayes u w (π n w) := fun _ w => Classical.choose_spec (exists_isBayes u w)
  exact le_trans (horizonWeight_mul_actNowValue_le_policyValue u he hδ hπ n w)
    (policyValue_le_adaptiveValue u e hδ π n w)

/-- The perfect-information value is additive over the signals of a normalised experiment. -/
theorem sum_perfectValue_updateWeights (u : A → Θ → K) {e : A → Θ → Z → K}
    (he : ∀ a, NormalisedKernel (e a)) (w : Θ → K) (a : A) :
    ∑ z, perfectValue u (updateWeights e w a z) = perfectValue u w := by
  unfold perfectValue updateWeights
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun θ _ => ?_
  rw [← Finset.sum_mul, ← Finset.mul_sum, he a θ, mul_one]

/-- No policy does better than knowing the state from the start. -/
theorem adaptiveValue_le_horizonWeight_mul_perfectValue (u : A → Θ → K) {e : A → Θ → Z → K}
    (he : ∀ a, NormalisedKernel (e a)) (hn : ∀ a, NonnegKernel (e a)) {δ : K} (hδ : 0 ≤ δ)
    (n : ℕ) {w : Θ → K} (hw : NonnegPrior w) :
    adaptiveValue u e δ n w ≤ horizonWeight δ n * perfectValue u w := by
  induction n generalizing w with
  | zero => simp [horizonWeight, adaptiveValue]
  | succ n ih =>
    refine maxOver_le fun a => ?_
    have h0 : expectedUtility u w a ≤ perfectValue u w :=
      Finset.sum_le_sum fun θ _ =>
        mul_le_mul_of_nonneg_left (le_maxOver (fun a => u a θ) a) (hw θ)
    have h1 : ∑ z, adaptiveValue u e δ n (updateWeights e w a z) ≤
        horizonWeight δ n * perfectValue u w := by
      calc ∑ z, adaptiveValue u e δ n (updateWeights e w a z)
          ≤ ∑ z, horizonWeight δ n * perfectValue u (updateWeights e w a z) :=
            Finset.sum_le_sum fun z _ => ih fun θ => mul_nonneg (hw θ) (hn a θ z)
        _ = horizonWeight δ n * ∑ z, perfectValue u (updateWeights e w a z) :=
            (Finset.mul_sum _ _ _).symm
        _ = _ := by rw [sum_perfectValue_updateWeights u he]
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    simp only [horizonWeight]
    linarith

end DecisionAnalysisProofs
