import DecisionAnalysisProofs.Finite.ValueOfInformation
import Mathlib.Tactic.FieldSimp

/-!
# DecisionAnalysisProofs.Finite.Posterior

Bayesian inversion and the two forms of the value with information (`src/experiments.jl`:
`bayes_invert`; `src/voi.jl`: `value_with_information`, `evsi` with the `Exact` method).

The definitions of `Finite/Problem.lean` use the **joint form**
`Σ_z max_a Σ_θ p θ · e θ z · u a θ`, which never divides. An implementation may instead
compute posteriors with `bayes_invert` and use the **posterior form**
`Σ_z P(z) · max_a E[u a | z]`, skipping signals of probability zero. This module proves that
the two agree, so either implementation is covered by the theorems about the joint form.

* `posterior p e z θ = p θ · e θ z / P(z)` (`bayes_invert`), with `P(z) = signalProbability`.
  In Lean `x / 0 = 0`, so the posterior of a zero-probability signal is the zero vector; no
  theorem below relies on that value except to say that the corresponding term vanishes.
* **`posterior_normalised`**: the posterior sums to one whenever `P(z) ≠ 0`.
  **`posterior_nonneg`**: it is nonnegative for a nonnegative prior and experiment.
* **`signalValue_eq_posterior`**: for `P(z) ≠ 0`, the joint-form objective is
  `P(z) · E[u a | z]`.
* **`signalValue_eq_zero`**, **`maxOver_signalValue_eq_zero`**: a signal of probability zero
  contributes nothing, whichever action is taken.
* **`valueWithInformation_eq_posterior_form`**: the joint form equals the posterior form.
* **`sum_signalProbability`**: the signal probabilities sum to the total prior mass.
* **`sum_signalProbability_mul_posterior`**: the posteriors average back to the prior.

Not claimed: any statement about posteriors computed in `Float64`, where a tiny positive
`P(z)` is a conditioning hazard that exact arithmetic does not have.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- `bayes_invert`: the posterior over states after signal `z`. -/
def posterior (p : Θ → K) (e : Θ → Z → K) (z : Z) : Θ → K :=
  fun θ => p θ * e θ z / signalProbability p e z

theorem signalProbability_nonneg {p : Θ → K} (hp : NonnegPrior p) {e : Θ → Z → K}
    (he : NonnegKernel e) (z : Z) : 0 ≤ signalProbability p e z :=
  Finset.sum_nonneg fun θ _ => mul_nonneg (hp θ) (he θ z)

/-- The signal probabilities sum to the total prior mass (one, for a normalised prior). -/
theorem sum_signalProbability (p : Θ → K) {e : Θ → Z → K} (he : NormalisedKernel e) :
    ∑ z, signalProbability p e z = ∑ θ, p θ := by
  unfold signalProbability
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun θ _ => by rw [← Finset.mul_sum, he θ, mul_one]

/-- `bayes_invert` returns a normalised posterior for every signal of nonzero probability. -/
theorem posterior_normalised (p : Θ → K) (e : Θ → Z → K) {z : Z}
    (h : signalProbability p e z ≠ 0) : ∑ θ, posterior p e z θ = 1 := by
  unfold posterior
  simp only [div_eq_mul_inv]
  rw [← Finset.sum_mul]
  exact mul_inv_cancel₀ h

theorem posterior_nonneg {p : Θ → K} (hp : NonnegPrior p) {e : Θ → Z → K}
    (he : NonnegKernel e) (z : Z) (θ : Θ) : 0 ≤ posterior p e z θ :=
  div_nonneg (mul_nonneg (hp θ) (he θ z)) (signalProbability_nonneg hp he z)

/-- The posteriors average back to the prior. -/
theorem sum_signalProbability_mul_posterior {p : Θ → K} (hp : NonnegPrior p) {e : Θ → Z → K}
    (he : StochasticKernel e) (θ : Θ) :
    ∑ z, signalProbability p e z * posterior p e z θ = p θ := by
  have hz : ∀ z, signalProbability p e z * posterior p e z θ = p θ * e θ z := by
    intro z
    by_cases h : signalProbability p e z = 0
    · have hnn : ∀ θ ∈ Finset.univ, 0 ≤ p θ * e θ z := fun θ _ =>
        mul_nonneg (hp θ) (he.1 θ z)
      have h0 := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp h
      rw [h, zero_mul, h0 θ (Finset.mem_univ θ)]
    · unfold posterior
      field_simp
  rw [Finset.sum_congr rfl fun z _ => hz z, ← Finset.mul_sum, he.2 θ, mul_one]

/-- For a signal of nonzero probability the joint-form objective is the probability of the
signal times the posterior expected utility. -/
theorem signalValue_eq_posterior (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) {z : Z}
    (h : signalProbability p e z ≠ 0) (a : A) :
    signalValue u p e z a =
      signalProbability p e z * expectedUtility u (posterior p e z) a := by
  unfold signalValue expectedUtility posterior
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun θ _ => ?_
  field_simp

/-- A signal of probability zero contributes nothing to the joint-form objective. -/
theorem signalValue_eq_zero (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p) {e : Θ → Z → K}
    (he : NonnegKernel e) {z : Z} (h : signalProbability p e z = 0) (a : A) :
    signalValue u p e z a = 0 := by
  have hnn : ∀ θ ∈ Finset.univ, 0 ≤ p θ * e θ z := fun θ _ => mul_nonneg (hp θ) (he θ z)
  have h0 := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp h
  exact Finset.sum_eq_zero fun θ hθ => by rw [h0 θ hθ, zero_mul]

theorem maxOver_signalValue_eq_zero (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p)
    {e : Θ → Z → K} (he : NonnegKernel e) {z : Z} (h : signalProbability p e z = 0) :
    maxOver (signalValue u p e z) = 0 := by
  obtain ⟨a, ha⟩ := exists_eq_maxOver (signalValue u p e z)
  rw [ha, signalValue_eq_zero u hp he h]

/-- One term of the value with information, in posterior form. -/
theorem maxOver_signalValue_eq_posterior (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p)
    {e : Θ → Z → K} (he : NonnegKernel e) (z : Z) :
    maxOver (signalValue u p e z) =
      signalProbability p e z * maxOver fun a => expectedUtility u (posterior p e z) a := by
  by_cases h : signalProbability p e z = 0
  · rw [maxOver_signalValue_eq_zero u hp he h, h, zero_mul]
  · have hf : signalValue u p e z =
        fun a => signalProbability p e z * expectedUtility u (posterior p e z) a :=
      funext (signalValue_eq_posterior u p e h)
    rw [hf, maxOver_mul_left _ (signalProbability_nonneg hp he z)]

/-- The joint (no-division) form of `value_with_information` equals the posterior form
`Σ_z P(z) · max_a E[u a | z]`; signals of probability zero contribute `0` to both. -/
theorem valueWithInformation_eq_posterior_form (u : A → Θ → K) {p : Θ → K}
    (hp : NonnegPrior p) {e : Θ → Z → K} (he : NonnegKernel e) :
    valueWithInformation u p e =
      ∑ z, signalProbability p e z * maxOver fun a => expectedUtility u (posterior p e z) a :=
  Finset.sum_congr rfl fun z _ => maxOver_signalValue_eq_posterior u hp he z

end DecisionAnalysisProofs
