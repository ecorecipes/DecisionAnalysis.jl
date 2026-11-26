import DecisionAnalysisProofs.Finite.AdaptiveDynamic

/-!
# DecisionAnalysisProofs.Finite.ValueVectors

The value of the best policy as a maximum of finitely many linear functions of the weights
(`adaptive_management` with `method = :vectors`, `src/adaptive.jl`). The recursion over
beliefs visits a number of beliefs that grows with every period; the same value function is
the upper envelope of a finite family of vectors, and over long horizons the family is far
smaller than the set of beliefs.

* `backupVector u k δ α a f`: the vector of "take `a` now, and after the signal `z` follow
  the vector `α (f z)`".

## What each result backs or shapes

* **`dynamicValue_succ_of_vectors`**: if the value with `n` periods to go is the maximum of
  `w ↦ ∑ θ, w θ * α i θ` over a family `α`, then the value with `n + 1` to go is the maximum
  over the vectors `backupVector u k δ α a f`, one for each action `a` and each choice `f` of
  a vector of the family for every signal. This is the step the package takes, and the reason
  it is exact.
* **`exists_vectors_dynamicValue`**: so for every horizon the value is such a maximum.
* **`sum_maxOver_eq_maxOver_sum`**: a sum of maxima is the maximum over choice functions of
  the sums; the interchange behind the step.
* **`le_maxOver_of_le_mixture`**: a number that is at most a mixture of the members of a
  family is at most their maximum. Backs the pruning: a vector that a mixture of the others
  matches or beats at every state is never needed, at any weights (apply it with the family
  of the values at `w`), so leaving it out changes nothing.

* **`sum_mul_backupVector_le_dynamicValue`**: vectors that are below the value stay below it
  when backed up, so the vectors kept at the points of a grid (`method = :points`) give a
  lower bound, `active`.
* **`dynamicValue_weighted_sum_le`** and **`dynamicValue_succ_le_of_upper`**: the value is
  sublinear in the weights, and an upper bound stays one when carried a step back. Together
  they back `active_upper`: a belief is written as a nonnegative combination of the corners
  and one grid point, and the bound is the same combination of their bounds.

Not claimed: that the pruned family is the smallest possible, how wide the gap between the two
bounds of `method = :points` is, or anything about the passive manager, whose value is not a
maximum of linear functions.
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- A sum of maxima is the maximum, over choice functions, of the sums. -/
theorem sum_maxOver_eq_maxOver_sum {ι : Type} [Fintype ι] [Nonempty ι] [DecidableEq Z]
    (g : Z → ι → K) : ∑ z, maxOver (g z) = maxOver fun f : Z → ι => ∑ z, g z (f z) := by
  classical
  refine le_antisymm ?_ (maxOver_le fun f => Finset.sum_le_sum fun z _ => le_maxOver (g z) (f z))
  let f : Z → ι := fun z => Classical.choose (exists_eq_maxOver (g z))
  have hf : ∀ z, maxOver (g z) = g z (f z) := fun z =>
    Classical.choose_spec (exists_eq_maxOver (g z))
  calc ∑ z, maxOver (g z) = ∑ z, g z (f z) := Finset.sum_congr rfl fun z _ => hf z
    _ ≤ _ := le_maxOver (fun f : Z → ι => ∑ z, g z (f z)) f

/-- A mixture of the members of a family is at most their maximum. -/
theorem mixture_le_maxOver {ι : Type} [Fintype ι] [Nonempty ι] (f : ι → K) {μ : ι → K}
    (hμ : ∀ i, 0 ≤ μ i) (h1 : ∑ i, μ i = 1) : ∑ i, μ i * f i ≤ maxOver f := by
  calc ∑ i, μ i * f i ≤ ∑ i, μ i * maxOver f :=
        Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (le_maxOver f i) (hμ i)
    _ = maxOver f := by rw [← Finset.sum_mul, h1, one_mul]

/-- What a mixture of a family matches or beats is at most the maximum of the family. -/
theorem le_maxOver_of_le_mixture {ι : Type} [Fintype ι] [Nonempty ι] (f : ι → K) {μ : ι → K}
    (hμ : ∀ i, 0 ≤ μ i) (h1 : ∑ i, μ i = 1) {c : K} (hc : c ≤ ∑ i, μ i * f i) :
    c ≤ maxOver f :=
  le_trans hc (mixture_le_maxOver f hμ h1)

/-- The vector of taking `a` now and, after the signal `z`, following the vector `α (f z)`. -/
def backupVector {ι : Type} (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K) (α : ι → Θ → K)
    (a : A) (f : Z → ι) : Θ → K :=
  fun θ => u a θ + δ * ∑ z, ∑ θ', k a z θ θ' * α (f z) θ'

/-- The value of a backed-up vector at `w`: the payoff of `a` now and the vectors chosen by
`f` at the weights of the next period. -/
theorem sum_mul_backupVector {ι : Type} (u : A → Θ → K) (k : A → Z → Θ → Θ → K) (δ : K)
    (α : ι → Θ → K) (a : A) (f : Z → ι) (w : Θ → K) :
    ∑ θ, w θ * backupVector u k δ α a f θ =
      expectedUtility u w a + δ * ∑ z, ∑ θ', stepWeights k w a z θ' * α (f z) θ' := by
  have h : ∑ z, ∑ θ', stepWeights k w a z θ' * α (f z) θ' =
      ∑ θ, w θ * ∑ z, ∑ θ', k a z θ θ' * α (f z) θ' := by
    unfold stepWeights
    simp only [Finset.sum_mul, Finset.mul_sum]
    refine Eq.trans (Finset.sum_congr rfl fun z _ => Finset.sum_comm) ?_
    refine Eq.trans Finset.sum_comm ?_
    refine Finset.sum_congr rfl fun θ _ => Finset.sum_congr rfl fun z _ =>
      Finset.sum_congr rfl fun θ' _ => ?_
    ring
  rw [h]
  unfold backupVector expectedUtility
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun θ _ => by ring

/-- One step of the recursion on vectors: from a family for `n` periods to the family of
backed-up vectors for `n + 1`. -/
theorem dynamicValue_succ_of_vectors {ι : Type} [Fintype ι] [Nonempty ι] [DecidableEq Z]
    (u : A → Θ → K) (k : A → Z → Θ → Θ → K) {δ : K} (hδ : 0 ≤ δ) (α : ι → Θ → K) (n : ℕ)
    (h : ∀ w, dynamicValue u k δ n w = maxOver fun i => ∑ θ, w θ * α i θ) (w : Θ → K) :
    dynamicValue u k δ (n + 1) w =
      maxOver fun p : A × (Z → ι) => ∑ θ, w θ * backupVector u k δ α p.1 p.2 θ := by
  have hstep : ∀ a, ∑ z, dynamicValue u k δ n (stepWeights k w a z) =
      maxOver fun f : Z → ι => ∑ z, ∑ θ', stepWeights k w a z θ' * α (f z) θ' := by
    intro a
    refine Eq.trans (Finset.sum_congr rfl fun z _ => h _) ?_
    exact sum_maxOver_eq_maxOver_sum fun z i => ∑ θ', stepWeights k w a z θ' * α i θ'
  refine le_antisymm (maxOver_le fun a => ?_) (maxOver_le fun p => ?_)
  · obtain ⟨f, hf⟩ := exists_eq_maxOver
      (fun f : Z → ι => ∑ z, ∑ θ', stepWeights k w a z θ' * α (f z) θ')
    rw [hstep a, hf, ← sum_mul_backupVector]
    exact le_maxOver (fun p : A × (Z → ι) => ∑ θ, w θ * backupVector u k δ α p.1 p.2 θ) (a, f)
  · rw [sum_mul_backupVector]
    refine le_trans ?_ (le_maxOver _ p.1)
    have h1 : ∑ z, ∑ θ', stepWeights k w p.1 z θ' * α (p.2 z) θ' ≤
        ∑ z, dynamicValue u k δ n (stepWeights k w p.1 z) := by
      rw [hstep p.1]
      exact le_maxOver (fun f : Z → ι => ∑ z, ∑ θ', stepWeights k w p.1 z θ' * α (f z) θ') p.2
    have h2 := mul_le_mul_of_nonneg_left h1 hδ
    linarith

/-- For every horizon the value of the best policy is the maximum of finitely many linear
functions of the weights. -/
theorem exists_vectors_dynamicValue (u : A → Θ → K) (k : A → Z → Θ → Θ → K) {δ : K}
    (hδ : 0 ≤ δ) (n : ℕ) :
    ∃ (ι : Type) (_ : Fintype ι) (_ : Nonempty ι) (α : ι → Θ → K),
      ∀ w, dynamicValue u k δ n w = maxOver fun i => ∑ θ, w θ * α i θ := by
  classical
  induction n with
  | zero =>
    refine ⟨Unit, inferInstance, inferInstance, fun _ _ => 0, fun w => ?_⟩
    rw [maxOver_unit]
    simp [dynamicValue]
  | succ n ih =>
    obtain ⟨ι, hF, hN, α, h⟩ := ih
    exact ⟨A × (Z → ι), inferInstance, inferInstance,
      fun p => backupVector u k δ α p.1 p.2,
      fun w => dynamicValue_succ_of_vectors u k hδ α n h w⟩

/-- A lower bound carried one step back. If every vector of a family is, at all weights, at
most the value with `n` periods to go, then every backed-up vector is at most the value with
`n + 1` to go: the vectors kept at chosen beliefs (`method = :points`) bound the value from
below, whichever vectors are left out. -/
theorem sum_mul_backupVector_le_dynamicValue {ι : Type} (u : A → Θ → K)
    (k : A → Z → Θ → Θ → K) {δ : K} (hδ : 0 ≤ δ) (α : ι → Θ → K) (n : ℕ)
    (h : ∀ i w, ∑ θ, w θ * α i θ ≤ dynamicValue u k δ n w) (a : A) (f : Z → ι) (w : Θ → K) :
    ∑ θ, w θ * backupVector u k δ α a f θ ≤ dynamicValue u k δ (n + 1) w := by
  rw [sum_mul_backupVector]
  refine le_trans ?_ (le_maxOver _ a)
  have h1 : ∑ z, ∑ θ', stepWeights k w a z θ' * α (f z) θ' ≤
      ∑ z, dynamicValue u k δ n (stepWeights k w a z) :=
    Finset.sum_le_sum fun z _ => h (f z) _
  have h2 := mul_le_mul_of_nonneg_left h1 hδ
  linarith

/-- The value is sublinear in the weights: at a nonnegative combination of weights it is at
most the same combination of the values. This is what makes an interpolation between beliefs
an upper bound (`active_upper` of `method = :points`). -/
theorem dynamicValue_weighted_sum_le {J : Type} [Fintype J] (u : A → Θ → K)
    (k : A → Z → Θ → Θ → K) {δ : K} (hδ : 0 ≤ δ) (n : ℕ) {c : J → K} (hc : ∀ j, 0 ≤ c j)
    (v : J → Θ → K) :
    dynamicValue u k δ n (fun θ => ∑ j, c j * v j θ) ≤ ∑ j, c j * dynamicValue u k δ n (v j) := by
  obtain ⟨ι, hF, hN, α, h⟩ := exists_vectors_dynamicValue u k hδ n
  rw [h]
  refine maxOver_le fun i => ?_
  have hsplit : ∑ θ, (∑ j, c j * v j θ) * α i θ = ∑ j, c j * ∑ θ, v j θ * α i θ := by
    simp only [Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun θ _ => by ring
  rw [hsplit]
  refine Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left ?_ (hc j)
  rw [h (v j)]
  exact le_maxOver (fun i => ∑ θ, v j θ * α i θ) i

/-- An upper bound carried one step back: a function that is at least the value with `n`
periods to go gives, by one step of the recursion, at least the value with `n + 1`. -/
theorem dynamicValue_succ_le_of_upper (u : A → Θ → K) (k : A → Z → Θ → Θ → K) {δ : K}
    (hδ : 0 ≤ δ) (n : ℕ) (U : (Θ → K) → K) (hU : ∀ w, dynamicValue u k δ n w ≤ U w)
    (w : Θ → K) :
    dynamicValue u k δ (n + 1) w ≤
      maxOver fun a => expectedUtility u w a + δ * ∑ z, U (stepWeights k w a z) := by
  refine maxOver_mono fun a => ?_
  have h1 : ∑ z, dynamicValue u k δ n (stepWeights k w a z) ≤ ∑ z, U (stepWeights k w a z) :=
    Finset.sum_le_sum fun z _ => hU _
  have h2 := mul_le_mul_of_nonneg_left h1 hδ
  linarith

end DecisionAnalysisProofs
