import DecisionAnalysisProofs.Finite.Adaptive

/-!
# DecisionAnalysisProofs.Finite.AdaptiveDynamic

Sequential management when the state **changes** from period to period
(`adaptive_management` with `transitions`, `src/adaptive.jl`). In a period the state is `θ`,
the action `a` pays `u a θ`, a signal `z` is seen and the state moves to `θ'`. All of that is
one kernel `k a z θ θ'`: the probability of the signal `z` and the next state `θ'`, given `θ`
and `a`. The two conventions of the package are instances (`signalThenMove`,
`moveThenSignal`), and a state that never moves is the instance treated in `Adaptive.lean`.

Weights are unnormalised, as everywhere: after `a` and `z` the weights `w` become
`stepWeights k w a z θ' = ∑ θ, w θ * k a z θ θ'`.

* `dynamicValue u k δ n w`: the value of the best policy (active adaptive management).
* `dynamicPolicyValue u k δ π n w`: the value of the policy `π`.
* `openLoopValue u k δ n w`: the value of the best plan fixed in advance, which allows for
  how the state moves but ignores every signal ("no learning").
* `observedValue u k δ n θ`: the value when the state is seen before each action.

## What each result backs or shapes

* **`openLoopValue_le_dynamicPolicyValue`**: a manager who replans in each period, taking the
  first action of the plan that is best for the current weights, and updates on what is seen
  (but never acts in order to learn) does at least as well as the best plan fixed in advance.
  Backs `passive ≥ no_learning`, and **shapes the package**: with a moving state the passive
  manager must replan over the remaining periods. One who only takes the best action for the
  current period can do worse than the fixed plan, so that definition would break the order.
* **`dynamicPolicyValue_le_dynamicValue`**: no policy beats the recursion. Backs
  `passive ≤ active`.
* **`openLoopValue_le_dynamicValue`**: backs `value_of_learning ≥ 0`.
* **`dynamicValue_le_sum_observedValue`**: no policy does better than seeing the state before
  every action. Backs `active ≤ perfect`, and shapes the package: with a moving state
  `perfect` is the state seen in every period, not only at the start.
* **`sum_signalThenMove`**, **`sum_moveThenSignal`**: summed over signals, either convention
  moves the state by the transition alone, so `perfect` and `no_learning` do not depend on the
  experiments.

Not claimed: rewards, transitions or experiments that change with the period, or anything
about how the recursion is memoised in the package.
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The weights after action `a` and signal `z`, moved to the next period, unnormalised. -/
def stepWeights (k : A → Z → Θ → Θ → K) (w : Θ → K) (a : A) (z : Z) : Θ → K :=
  fun θ' => ∑ θ, w θ * k a z θ θ'

/-- The weights moved to the next period under `a` when no signal is used. -/
def pushWeights (k : A → Z → Θ → Θ → K) (w : Θ → K) (a : A) : Θ → K :=
  fun θ' => ∑ z, stepWeights k w a z θ'

/-- The signal is of the state of the period, which then moves. -/
def signalThenMove (e : A → Θ → Z → K) (t : A → Θ → Θ → K) : A → Z → Θ → Θ → K :=
  fun a z θ θ' => e a θ z * t a θ θ'

/-- The state moves, and the signal is of the new state. -/
def moveThenSignal (e : A → Θ → Z → K) (t : A → Θ → Θ → K) : A → Z → Θ → Θ → K :=
  fun a z θ θ' => t a θ θ' * e a θ' z

theorem sum_signalThenMove {e : A → Θ → Z → K} (he : ∀ a, NormalisedKernel (e a))
    (t : A → Θ → Θ → K) (a : A) (θ θ' : Θ) : ∑ z, signalThenMove e t a z θ θ' = t a θ θ' := by
  unfold signalThenMove
  rw [← Finset.sum_mul, he a θ, one_mul]

theorem sum_moveThenSignal {e : A → Θ → Z → K} (he : ∀ a, NormalisedKernel (e a))
    (t : A → Θ → Θ → K) (a : A) (θ θ' : Θ) : ∑ z, moveThenSignal e t a z θ θ' = t a θ θ' := by
  unfold moveThenSignal
  rw [← Finset.mul_sum, he a θ', mul_one]

/-- The value of the best policy over `n` periods from the weights `w`. -/
def dynamicValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K) : ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => maxOver fun a =>
      expectedUtility u w a + δ * ∑ z, dynamicValue u k δ n (stepWeights k w a z)

/-- The value of the policy `π` over `n` periods from the weights `w`. -/
def dynamicPolicyValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K)
    (π : ℕ → (Θ → K) → A) : ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => expectedUtility u w (π n w) +
      δ * ∑ z, dynamicPolicyValue u k δ π n (stepWeights k w (π n w) z)

/-- The value of the best plan fixed in advance: the state moves, no signal is used. -/
def openLoopValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K) : ℕ → (Θ → K) → K
  | 0, _ => 0
  | n + 1, w => maxOver fun a =>
      expectedUtility u w a + δ * openLoopValue u k δ n (pushWeights k w a)

/-- The value over `n` periods from the state `θ` when the state is seen before each action. -/
def observedValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K) : ℕ → Θ → K
  | 0, _ => 0
  | n + 1, θ => maxOver fun a =>
      u a θ + δ * ∑ z, ∑ θ', k a z θ θ' * observedValue u k δ n θ'

/-- No policy is worth more than the recursion. -/
theorem dynamicPolicyValue_le_dynamicValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) {δ : K}
    (hδ : 0 ≤ δ) (π : ℕ → (Θ → K) → A) (n : ℕ) (w : Θ → K) :
    dynamicPolicyValue u k δ π n w ≤ dynamicValue u k δ n w := by
  induction n generalizing w with
  | zero => exact le_refl _
  | succ n ih =>
    refine le_trans ?_ (le_maxOver _ (π n w))
    have h := mul_le_mul_of_nonneg_left
      (Finset.sum_le_sum fun z (_ : z ∈ Finset.univ) =>
        ih (stepWeights k w (π n w) z)) hδ
    show expectedUtility u w (π n w) +
        δ * ∑ z, dynamicPolicyValue u k δ π n (stepWeights k w (π n w) z) ≤
      expectedUtility u w (π n w) +
        δ * ∑ z, dynamicValue u k δ n (stepWeights k w (π n w) z)
    linarith

/-- The value of a fixed plan from a sum of weights is at most the sum of the values: a plan
chosen separately for each part does at least as well as one plan for all. -/
theorem openLoopValue_sum_le (u : A → Θ → K) (k : A → Z → Θ → Θ → K) {δ : K} (hδ : 0 ≤ δ)
    (n : ℕ) (v : Z → Θ → K) :
    openLoopValue u k δ n (fun θ => ∑ z, v z θ) ≤ ∑ z, openLoopValue u k δ n (v z) := by
  induction n generalizing v with
  | zero => simp [openLoopValue]
  | succ n ih =>
    refine maxOver_le fun a => ?_
    have hu : expectedUtility u (fun θ => ∑ z, v z θ) a = ∑ z, expectedUtility u (v z) a := by
      unfold expectedUtility
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun θ _ => ?_
      rw [Finset.sum_mul]
    have hp : pushWeights k (fun θ => ∑ z, v z θ) a = fun θ' => ∑ z, pushWeights k (v z) a θ' := by
      funext θ'
      unfold pushWeights stepWeights
      simp only [Finset.sum_mul]
      refine Eq.trans (Finset.sum_congr rfl fun z _ => Finset.sum_comm) ?_
      exact Finset.sum_comm
    have h1 : openLoopValue u k δ n (pushWeights k (fun θ => ∑ z, v z θ) a) ≤
        ∑ z, openLoopValue u k δ n (pushWeights k (v z) a) := by
      rw [hp]; exact ih _
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    have h3 : ∑ z, (expectedUtility u (v z) a +
        δ * openLoopValue u k δ n (pushWeights k (v z) a)) ≤
        ∑ z, openLoopValue u k δ (n + 1) (v z) :=
      Finset.sum_le_sum fun z _ => le_maxOver
        (fun a => expectedUtility u (v z) a + δ * openLoopValue u k δ n (pushWeights k (v z) a)) a
    rw [Finset.sum_add_distrib, ← Finset.mul_sum] at h3
    rw [hu]
    linarith

/-- A manager who takes the first action of the best fixed plan for the current weights, and
replans after every signal, does at least as well as the best fixed plan. -/
theorem openLoopValue_le_dynamicPolicyValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) {δ : K}
    (hδ : 0 ≤ δ) {π : ℕ → (Θ → K) → A}
    (hπ : ∀ n w, openLoopValue u k δ (n + 1) w =
      expectedUtility u w (π n w) + δ * openLoopValue u k δ n (pushWeights k w (π n w)))
    (n : ℕ) (w : Θ → K) :
    openLoopValue u k δ n w ≤ dynamicPolicyValue u k δ π n w := by
  induction n generalizing w with
  | zero => exact le_refl _
  | succ n ih =>
    have h1 : openLoopValue u k δ n (pushWeights k w (π n w)) ≤
        ∑ z, dynamicPolicyValue u k δ π n (stepWeights k w (π n w) z) :=
      le_trans (openLoopValue_sum_le u k hδ n fun z => stepWeights k w (π n w) z)
        (Finset.sum_le_sum fun z _ => ih _)
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    rw [hπ n w]
    show expectedUtility u w (π n w) + δ * openLoopValue u k δ n (pushWeights k w (π n w)) ≤
      expectedUtility u w (π n w) +
        δ * ∑ z, dynamicPolicyValue u k δ π n (stepWeights k w (π n w) z)
    linarith

/-- The active value is at least the value of the best fixed plan. -/
theorem openLoopValue_le_dynamicValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) {δ : K}
    (hδ : 0 ≤ δ) (n : ℕ) (w : Θ → K) :
    openLoopValue u k δ n w ≤ dynamicValue u k δ n w := by
  classical
  let f : ℕ → (Θ → K) → A → K := fun n w a =>
    expectedUtility u w a + δ * openLoopValue u k δ n (pushWeights k w a)
  let π : ℕ → (Θ → K) → A := fun n w => Classical.choose (exists_eq_maxOver (f n w))
  have hπ : ∀ n w, maxOver (f n w) = f n w (π n w) := fun n w =>
    Classical.choose_spec (exists_eq_maxOver (f n w))
  exact le_trans (openLoopValue_le_dynamicPolicyValue u k hδ (π := π) hπ n w)
    (dynamicPolicyValue_le_dynamicValue u k hδ π n w)

/-- No policy does better than seeing the state before every action. -/
theorem dynamicValue_le_sum_observedValue (u : A → Θ → K) {k : A → Z → Θ → Θ → K}
    (hk : ∀ a z θ θ', 0 ≤ k a z θ θ') {δ : K} (hδ : 0 ≤ δ) (n : ℕ) {w : Θ → K}
    (hw : NonnegPrior w) :
    dynamicValue u k δ n w ≤ ∑ θ, w θ * observedValue u k δ n θ := by
  induction n generalizing w with
  | zero => simp [dynamicValue, observedValue]
  | succ n ih =>
    refine maxOver_le fun a => ?_
    have h1 : ∑ z, dynamicValue u k δ n (stepWeights k w a z) ≤
        ∑ θ, w θ * ∑ z, ∑ θ', k a z θ θ' * observedValue u k δ n θ' := by
      calc ∑ z, dynamicValue u k δ n (stepWeights k w a z)
          ≤ ∑ z, ∑ θ', stepWeights k w a z θ' * observedValue u k δ n θ' :=
            Finset.sum_le_sum fun z _ => ih fun θ' =>
              Finset.sum_nonneg fun θ _ => mul_nonneg (hw θ) (hk a z θ θ')
        _ = _ := by
            unfold stepWeights
            simp only [Finset.sum_mul, Finset.mul_sum]
            refine Eq.trans (Finset.sum_congr rfl fun z _ => Finset.sum_comm) ?_
            refine Eq.trans Finset.sum_comm ?_
            refine Finset.sum_congr rfl fun θ _ => Finset.sum_congr rfl fun z _ =>
              Finset.sum_congr rfl fun θ' _ => ?_
            ring
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    have h3 : ∑ θ, w θ * (u a θ + δ * ∑ z, ∑ θ', k a z θ θ' * observedValue u k δ n θ') ≤
        ∑ θ, w θ * observedValue u k δ (n + 1) θ :=
      Finset.sum_le_sum fun θ _ => mul_le_mul_of_nonneg_left
        (le_maxOver (fun a => u a θ + δ * ∑ z, ∑ θ', k a z θ θ' * observedValue u k δ n θ') a)
        (hw θ)
    have h4 : ∑ θ, w θ * (u a θ + δ * ∑ z, ∑ θ', k a z θ θ' * observedValue u k δ n θ') =
        expectedUtility u w a +
          δ * ∑ θ, w θ * ∑ z, ∑ θ', k a z θ θ' * observedValue u k δ n θ' := by
      unfold expectedUtility
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun θ _ => by ring
    linarith

end DecisionAnalysisProofs
