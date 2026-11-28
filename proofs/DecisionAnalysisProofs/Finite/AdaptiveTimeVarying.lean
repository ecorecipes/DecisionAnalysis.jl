import DecisionAnalysisProofs.Finite.AdaptiveDynamic

/-!
# DecisionAnalysisProofs.Finite.AdaptiveTimeVarying

Time-varying sequential adaptive management (`adaptive_management` with per-period `payoffs`,
`experiments` and `transitions` schedules in `src/adaptive.jl`, tested in `test/test_track3.jl`)
and positive homogeneity of the Bellman recursion (backing belief normalisation in `_best` /
`_normalise_belief`, Track 2 O2).

## Conventions

* At `n + 1` periods remaining, the current period uses payoff table `u n : A → Θ → K` and
  joint signal-and-transition kernel `k n : A → Z → Θ → Θ → K` (matching `_Q_at(m, n)` and
  `_updates(m, n, ...)` in `src/adaptive.jl`).
* `tvDynamicValue u k δ n w`: best active policy value over `n` remaining periods.
* `tvDynamicPolicyValue u k δ π n w`: value of policy `π` over `n` remaining periods.
* `tvOpenLoopValue u k δ n w`: best fixed open-loop plan over `n` remaining periods (no signals).
* `tvObservedValue u k δ n θ`: value when the state is observed before each action.

## Headline results

* **`tvDynamicPolicyValue_le_tvDynamicValue`**,
  **`exists_tvDynamicPolicyValue_eq_tvDynamicValue`**: the time-varying recursion is attained by
  a policy and dominates every policy (`passive ≤ active`, `value_of_probing ≥ 0`).
* **`tvOpenLoopValue_le_tvDynamicPolicyValue`**, **`tvOpenLoopValue_le_tvDynamicValue`**:
  replanning with updating dominates the best fixed plan (`no_learning ≤ passive ≤ active`,
  `value_of_learning ≥ 0`).
* **`tvDynamicValue_le_sum_tvObservedValue`**: no policy beats observing the state before every
  action (`active ≤ perfect`).
* **`tvDynamicValue_const_eq_dynamicValue`**, **`tvOpenLoopValue_const_eq_openLoopValue`**,
  **`tvObservedValue_const_eq_observedValue`**: constant schedules reduce identically to
  `AdaptiveDynamic.lean`.
* **`tvDynamicValue_smul`**, **`tvDynamicValue_eq_total_mul_normalised`**,
  **`tvDynamicValue_zero`**, **`dynamicValue_smul`**: positive homogeneity of degree 1 in the
  unnormalised belief vector `w`, certifying that normalising `w` by `total = ∑ w` before memo
  lookup (`_normalise_belief` in `_best`) preserves the exact value and optimal action.

Not claimed: floating-point key rounding (`_weights_key`) or grid approximation error for
`method = :points`.
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### Time-varying value recursions -/

/-- The value of the best active policy over `n` periods remaining with period-indexed payoffs
`u` and transition-observation kernels `k`. -/
def tvDynamicValue (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K) (δ : K) :
    ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => maxOver fun a =>
      expectedUtility (u n) w a + δ * ∑ z, tvDynamicValue u k δ n (stepWeights (k n) w a z)

/-- The value of policy `π` over `n` periods remaining under time-varying `u`, `k`. -/
def tvDynamicPolicyValue (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K) (δ : K)
    (π : ℕ → (Θ → K) → A) : ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => expectedUtility (u n) w (π n w) +
      δ * ∑ z, tvDynamicPolicyValue u k δ π n (stepWeights (k n) w (π n w) z)

/-- The value of the best fixed open-loop plan over `n` periods remaining (signals ignored). -/
def tvOpenLoopValue (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K) (δ : K) :
    ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => maxOver fun a =>
      expectedUtility (u n) w a + δ * tvOpenLoopValue u k δ n (pushWeights (k n) w a)

/-- The fully observed (perfect-information) value over `n` periods remaining from state `θ`. -/
def tvObservedValue (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K) (δ : K) : ℕ → Θ → K
  | 0, _ => 0
  | n + 1, θ => maxOver fun a =>
      u n a θ + δ * ∑ z, ∑ θ', k n a z θ θ' * tvObservedValue u k δ n θ'

/-! ### Ordering: `no_learning ≤ passive ≤ active ≤ perfect` -/

/-- No policy is worth more than the time-varying Bellman recursion. -/
theorem tvDynamicPolicyValue_le_tvDynamicValue (u : ℕ → A → Θ → K)
    (k : ℕ → A → Z → Θ → Θ → K) {δ : K} (hδ : 0 ≤ δ)
    (π : ℕ → (Θ → K) → A) (n : ℕ) (w : Θ → K) :
    tvDynamicPolicyValue u k δ π n w ≤ tvDynamicValue u k δ n w := by
  induction n generalizing w with
  | zero => exact le_refl _
  | succ n ih =>
    refine le_trans ?_ (le_maxOver _ (π n w))
    have h := mul_le_mul_of_nonneg_left
      (Finset.sum_le_sum fun z (_ : z ∈ Finset.univ) =>
        ih (stepWeights (k n) w (π n w) z)) hδ
    show expectedUtility (u n) w (π n w) +
        δ * ∑ z, tvDynamicPolicyValue u k δ π n (stepWeights (k n) w (π n w) z) ≤
      expectedUtility (u n) w (π n w) +
        δ * ∑ z, tvDynamicValue u k δ n (stepWeights (k n) w (π n w) z)
    linarith

/-- The time-varying recursion is attained by a policy. -/
theorem exists_tvDynamicPolicyValue_eq_tvDynamicValue (u : ℕ → A → Θ → K)
    (k : ℕ → A → Z → Θ → Θ → K) (δ : K) :
    ∃ π : ℕ → (Θ → K) → A, ∀ n w,
      tvDynamicPolicyValue u k δ π n w = tvDynamicValue u k δ n w := by
  classical
  let f : ℕ → (Θ → K) → A → K := fun n w a =>
    expectedUtility (u n) w a + δ * ∑ z, tvDynamicValue u k δ n (stepWeights (k n) w a z)
  let π : ℕ → (Θ → K) → A := fun n w => Classical.choose (exists_eq_maxOver (f n w))
  have hπ : ∀ n w, maxOver (f n w) = f n w (π n w) := fun n w =>
    Classical.choose_spec (exists_eq_maxOver (f n w))
  refine ⟨π, fun n => ?_⟩
  induction n with
  | zero => intro w; rfl
  | succ n ih =>
    intro w
    have step : tvDynamicPolicyValue u k δ π (n + 1) w = f n w (π n w) := by
      show expectedUtility (u n) w (π n w) +
          δ * ∑ z, tvDynamicPolicyValue u k δ π n (stepWeights (k n) w (π n w) z) =
        expectedUtility (u n) w (π n w) +
          δ * ∑ z, tvDynamicValue u k δ n (stepWeights (k n) w (π n w) z)
      rw [Finset.sum_congr rfl fun z _ => ih (stepWeights (k n) w (π n w) z)]
    exact step.trans (hπ n w).symm

/-- Subadditivity of the open-loop value in the unnormalised belief weights. -/
theorem tvOpenLoopValue_sum_le (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K)
    {δ : K} (hδ : 0 ≤ δ) (n : ℕ) (v : Z → Θ → K) :
    tvOpenLoopValue u k δ n (fun θ => ∑ z, v z θ) ≤ ∑ z, tvOpenLoopValue u k δ n (v z) := by
  induction n generalizing v with
  | zero => simp [tvOpenLoopValue]
  | succ n ih =>
    refine maxOver_le fun a => ?_
    have hu : expectedUtility (u n) (fun θ => ∑ z, v z θ) a =
        ∑ z, expectedUtility (u n) (v z) a := by
      unfold expectedUtility
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun θ _ => Finset.sum_mul _ _ _
    have hp : pushWeights (k n) (fun θ => ∑ z, v z θ) a =
        fun θ' => ∑ z, pushWeights (k n) (v z) a θ' := by
      funext θ'
      unfold pushWeights stepWeights
      simp only [Finset.sum_mul]
      exact (Finset.sum_congr rfl fun z _ => Finset.sum_comm).trans Finset.sum_comm
    have h1 : tvOpenLoopValue u k δ n (pushWeights (k n) (fun θ => ∑ z, v z θ) a) ≤
        ∑ z, tvOpenLoopValue u k δ n (pushWeights (k n) (v z) a) := by
      rw [hp]; exact ih _
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    have h3 : ∑ z, (expectedUtility (u n) (v z) a +
        δ * tvOpenLoopValue u k δ n (pushWeights (k n) (v z) a)) ≤
        ∑ z, tvOpenLoopValue u k δ (n + 1) (v z) :=
      Finset.sum_le_sum fun z _ => le_maxOver
        (fun a => expectedUtility (u n) (v z) a +
          δ * tvOpenLoopValue u k δ n (pushWeights (k n) (v z) a)) a
    rw [Finset.sum_add_distrib, ← Finset.mul_sum] at h3
    rw [hu]
    linarith

/-- Passive replanning with updating does at least as well as the best fixed open-loop plan. -/
theorem tvOpenLoopValue_le_tvDynamicPolicyValue (u : ℕ → A → Θ → K)
    (k : ℕ → A → Z → Θ → Θ → K) {δ : K} (hδ : 0 ≤ δ) {π : ℕ → (Θ → K) → A}
    (hπ : ∀ n w, tvOpenLoopValue u k δ (n + 1) w =
      expectedUtility (u n) w (π n w) +
        δ * tvOpenLoopValue u k δ n (pushWeights (k n) w (π n w)))
    (n : ℕ) (w : Θ → K) :
    tvOpenLoopValue u k δ n w ≤ tvDynamicPolicyValue u k δ π n w := by
  induction n generalizing w with
  | zero => exact le_refl _
  | succ n ih =>
    have h1 : tvOpenLoopValue u k δ n (pushWeights (k n) w (π n w)) ≤
        ∑ z, tvDynamicPolicyValue u k δ π n (stepWeights (k n) w (π n w) z) :=
      le_trans (tvOpenLoopValue_sum_le u k hδ n fun z => stepWeights (k n) w (π n w) z)
        (Finset.sum_le_sum fun z _ => ih _)
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    rw [hπ n w]
    show expectedUtility (u n) w (π n w) +
        δ * tvOpenLoopValue u k δ n (pushWeights (k n) w (π n w)) ≤
      expectedUtility (u n) w (π n w) +
        δ * ∑ z, tvDynamicPolicyValue u k δ π n (stepWeights (k n) w (π n w) z)
    linarith

/-- The active value is at least the best fixed plan (`value_of_learning ≥ 0`). -/
theorem tvOpenLoopValue_le_tvDynamicValue (u : ℕ → A → Θ → K)
    (k : ℕ → A → Z → Θ → Θ → K) {δ : K} (hδ : 0 ≤ δ) (n : ℕ) (w : Θ → K) :
    tvOpenLoopValue u k δ n w ≤ tvDynamicValue u k δ n w := by
  classical
  let f : ℕ → (Θ → K) → A → K := fun n w a =>
    expectedUtility (u n) w a + δ * tvOpenLoopValue u k δ n (pushWeights (k n) w a)
  let π : ℕ → (Θ → K) → A := fun n w => Classical.choose (exists_eq_maxOver (f n w))
  have hπ : ∀ n w, maxOver (f n w) = f n w (π n w) := fun n w =>
    Classical.choose_spec (exists_eq_maxOver (f n w))
  exact le_trans (tvOpenLoopValue_le_tvDynamicPolicyValue u k hδ (π := π) hπ n w)
    (tvDynamicPolicyValue_le_tvDynamicValue u k hδ π n w)

/-- No policy does better than seeing the state before every action (`active ≤ perfect`). -/
theorem tvDynamicValue_le_sum_tvObservedValue (u : ℕ → A → Θ → K)
    {k : ℕ → A → Z → Θ → Θ → K} (hk : ∀ n a z θ θ', 0 ≤ k n a z θ θ')
    {δ : K} (hδ : 0 ≤ δ) (n : ℕ) {w : Θ → K} (hw : NonnegPrior w) :
    tvDynamicValue u k δ n w ≤ ∑ θ, w θ * tvObservedValue u k δ n θ := by
  induction n generalizing w with
  | zero => simp [tvDynamicValue, tvObservedValue]
  | succ n ih =>
    refine maxOver_le fun a => ?_
    have h1 : ∑ z, tvDynamicValue u k δ n (stepWeights (k n) w a z) ≤
        ∑ θ, w θ * ∑ z, ∑ θ', k n a z θ θ' * tvObservedValue u k δ n θ' := by
      calc ∑ z, tvDynamicValue u k δ n (stepWeights (k n) w a z)
          ≤ ∑ z, ∑ θ', stepWeights (k n) w a z θ' * tvObservedValue u k δ n θ' :=
            Finset.sum_le_sum fun z _ => ih fun θ' =>
              Finset.sum_nonneg fun θ _ => mul_nonneg (hw θ) (hk n a z θ θ')
        _ = _ := by
            unfold stepWeights
            simp only [Finset.sum_mul, Finset.mul_sum]
            refine Eq.trans (Finset.sum_congr rfl fun z _ => Finset.sum_comm) ?_
            refine Eq.trans Finset.sum_comm ?_
            refine Finset.sum_congr rfl fun θ _ => Finset.sum_congr rfl fun z _ =>
              Finset.sum_congr rfl fun θ' _ => ?_
            ring
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    have h3 : ∑ θ, w θ * (u n a θ + δ * ∑ z, ∑ θ', k n a z θ θ' * tvObservedValue u k δ n θ') ≤
        ∑ θ, w θ * tvObservedValue u k δ (n + 1) θ :=
      Finset.sum_le_sum fun θ _ => mul_le_mul_of_nonneg_left
        (le_maxOver
          (fun a => u n a θ + δ * ∑ z, ∑ θ', k n a z θ θ' * tvObservedValue u k δ n θ') a)
        (hw θ)
    have h4 : ∑ θ, w θ * (u n a θ + δ * ∑ z, ∑ θ', k n a z θ θ' * tvObservedValue u k δ n θ') =
        expectedUtility (u n) w a +
          δ * ∑ θ, w θ * ∑ z, ∑ θ', k n a z θ θ' * tvObservedValue u k δ n θ' := by
      unfold expectedUtility
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun θ _ => by ring
    linarith

/-! ### Reduction to stationary `AdaptiveDynamic` -/

theorem tvDynamicValue_const_eq_dynamicValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K)
    (n : ℕ) (w : Θ → K) :
    tvDynamicValue (fun _ => u) (fun _ => k) δ n w = dynamicValue u k δ n w := by
  induction n generalizing w with
  | zero => rfl
  | succ n ih =>
    unfold tvDynamicValue dynamicValue
    refine congrArg maxOver (funext fun a => ?_)
    rw [Finset.sum_congr rfl fun z _ => ih (stepWeights k w a z)]

theorem tvOpenLoopValue_const_eq_openLoopValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K)
    (n : ℕ) (w : Θ → K) :
    tvOpenLoopValue (fun _ => u) (fun _ => k) δ n w = openLoopValue u k δ n w := by
  induction n generalizing w with
  | zero => rfl
  | succ n ih =>
    unfold tvOpenLoopValue openLoopValue
    refine congrArg maxOver (funext fun a => ?_)
    rw [ih]

theorem tvObservedValue_const_eq_observedValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K)
    (n : ℕ) (θ : Θ) :
    tvObservedValue (fun _ => u) (fun _ => k) δ n θ = observedValue u k δ n θ := by
  induction n generalizing θ with
  | zero => rfl
  | succ n ih =>
    unfold tvObservedValue observedValue
    refine congrArg maxOver (funext fun a => ?_)
    congr 2
    exact Finset.sum_congr rfl fun z _ =>
      Finset.sum_congr rfl fun θ' _ => by rw [ih θ']

/-! ### Positive homogeneity in the belief weights (backs O2 belief normalisation) -/

private theorem stepWeights_smul (k : A → Z → Θ → Θ → K) (c : K) (w : Θ → K) (a : A) (z : Z) :
    stepWeights k (fun θ => c * w θ) a z = fun θ' => c * stepWeights k w a z θ' := by
  funext θ'
  unfold stepWeights
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun θ _ => by ring

private theorem pushWeights_smul (k : A → Z → Θ → Θ → K) (c : K) (w : Θ → K) (a : A) :
    pushWeights k (fun θ => c * w θ) a = fun θ' => c * pushWeights k w a θ' := by
  funext θ'
  unfold pushWeights
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun z _ => by rw [stepWeights_smul]

private theorem expectedUtility_smul_prior (u : A → Θ → K) (c : K) (w : Θ → K) (a : A) :
    expectedUtility u (fun θ => c * w θ) a = c * expectedUtility u w a := by
  unfold expectedUtility
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun θ _ => by ring

/-- Positive homogeneity of degree 1: scaling unnormalised belief weights by `c ≥ 0` scales the
time-varying active value by `c`. Backs belief normalisation in `_best` (`src/adaptive.jl`). -/
theorem tvDynamicValue_smul (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K) (δ : K)
    (n : ℕ) {c : K} (hc : 0 ≤ c) (w : Θ → K) :
    tvDynamicValue u k δ n (fun θ => c * w θ) = c * tvDynamicValue u k δ n w := by
  induction n generalizing w with
  | zero => simp [tvDynamicValue]
  | succ n ih =>
    unfold tvDynamicValue
    rw [← maxOver_mul_left _ hc]
    refine congrArg maxOver (funext fun a => ?_)
    rw [expectedUtility_smul_prior]
    have hsum : ∑ z, tvDynamicValue u k δ n (stepWeights (k n) (fun θ => c * w θ) a z) =
        c * ∑ z, tvDynamicValue u k δ n (stepWeights (k n) w a z) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun z _ => ?_
      rw [stepWeights_smul, ih]
    rw [hsum]
    ring

/-- Normalising positive-total weights `w` by `total = ∑ w` and multiplying the memoised value
back by `total` yields the exact unnormalised value (`_normalise_belief` in `_best`). -/
theorem tvDynamicValue_eq_total_mul_normalised (u : ℕ → A → Θ → K)
    (k : ℕ → A → Z → Θ → Θ → K) (δ : K) (n : ℕ) (w : Θ → K) {total : K} (ht : 0 < total) :
    tvDynamicValue u k δ n w = total * tvDynamicValue u k δ n (fun θ => w θ / total) := by
  have hw : w = fun θ => total * (w θ / total) :=
    funext fun θ => (mul_div_cancel₀ _ (ne_of_gt ht)).symm
  conv_lhs => rw [hw]
  exact tvDynamicValue_smul u k δ n ht.le (fun θ => w θ / total)

/-- Zero unnormalised weights have value zero. -/
theorem tvDynamicValue_zero (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K) (δ : K)
    (n : ℕ) : tvDynamicValue u k δ n (fun _ => 0) = 0 := by
  have h := tvDynamicValue_smul u k δ n (le_refl (0 : K)) (fun _ => 0)
  simpa using h

/-- Positive homogeneity of degree 1 for stationary `dynamicValue`. -/
theorem dynamicValue_smul (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K)
    (n : ℕ) {c : K} (hc : 0 ≤ c) (w : Θ → K) :
    dynamicValue u k δ n (fun θ => c * w θ) = c * dynamicValue u k δ n w := by
  rw [← tvDynamicValue_const_eq_dynamicValue, ← tvDynamicValue_const_eq_dynamicValue]
  exact tvDynamicValue_smul (fun _ => u) (fun _ => k) δ n hc w

/-- Positive homogeneity of degree 1 for the time-varying open-loop value. -/
theorem tvOpenLoopValue_smul (u : ℕ → A → Θ → K) (k : ℕ → A → Z → Θ → Θ → K) (δ : K)
    (n : ℕ) {c : K} (hc : 0 ≤ c) (w : Θ → K) :
    tvOpenLoopValue u k δ n (fun θ => c * w θ) = c * tvOpenLoopValue u k δ n w := by
  induction n generalizing w with
  | zero => simp [tvOpenLoopValue]
  | succ n ih =>
    unfold tvOpenLoopValue
    rw [← maxOver_mul_left _ hc]
    refine congrArg maxOver (funext fun a => ?_)
    rw [expectedUtility_smul_prior, pushWeights_smul, ih]
    ring

end DecisionAnalysisProofs
