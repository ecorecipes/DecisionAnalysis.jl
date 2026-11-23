import DecisionAnalysisProofs.Finite.Problem

/-!
# DecisionAnalysisProofs.Finite.Affine

**T7**: positive affine rescaling of the payoff table (`src/utilities.jl`, `src/voi.jl`).
The rescaled table is `rescale s b u a θ = s · u a θ + b θ`; the shift `b` may depend on the
state, which is the strongest form under which the conclusions hold (a constant shift is the
special case `b = fun _ => c`).

* **`isBayes_rescale`**: for `0 < s` the Bayes action set is unchanged
  (`decide(problem, MaxExpectedUtility())`).
* **`evpi_rescale`**, **`evsi_rescale`**, **`evppi_rescale`**: for `0 ≤ s` each value of
  information is multiplied by `s`; the shift cancels. `evsi_rescale` needs the experiment to
  be normalised.
* **`regret_rescale`**, **`isMinimaxRegret_rescale`**: the regret table is multiplied by `s`
  and the minimax-regret action set is unchanged.
* Supporting identities: `expectedUtility_rescale`, `actNowValue_rescale`,
  `perfectValue_rescale`, `valueWithInformation_rescale`.

No hypothesis on the prior is needed. Not claimed: invariance under a shift that depends on the
action, which is false.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z C : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z] [Fintype C]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The payoff table `s · u + b`, with slope `s` and state-dependent shift `b`. -/
def rescale (s : K) (b : Θ → K) (u : A → Θ → K) : A → Θ → K := fun a θ => s * u a θ + b θ

theorem expectedUtility_rescale (s : K) (b : Θ → K) (u : A → Θ → K) (p : Θ → K) (a : A) :
    expectedUtility (rescale s b u) p a = s * expectedUtility u p a + ∑ θ, p θ * b θ := by
  unfold expectedUtility rescale
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun θ _ => by ring

/-- **T7.** A positive affine rescaling preserves the Bayes action set. -/
theorem isBayes_rescale {s : K} (hs : 0 < s) (b : Θ → K) (u : A → Θ → K) (p : Θ → K) (a : A) :
    IsBayes (rescale s b u) p a ↔ IsBayes u p a := by
  unfold IsBayes
  simp only [expectedUtility_rescale]
  constructor
  · intro h a'
    have h' := h a'
    exact le_of_mul_le_mul_left (by linarith) hs
  · intro h a'
    have h' := mul_le_mul_of_nonneg_left (h a') hs.le
    linarith

theorem actNowValue_rescale {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K) (p : Θ → K) :
    actNowValue (rescale s b u) p = s * actNowValue u p + ∑ θ, p θ * b θ := by
  have h : expectedUtility (rescale s b u) p =
      fun a => s * expectedUtility u p a + ∑ θ, p θ * b θ :=
    funext (expectedUtility_rescale s b u p)
  unfold actNowValue
  rw [h, maxOver_mul_add _ hs]

theorem perfectValue_rescale {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K) (p : Θ → K) :
    perfectValue (rescale s b u) p = s * perfectValue u p + ∑ θ, p θ * b θ := by
  simp only [perfectValue, rescale, maxOver_mul_add _ hs]
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun θ _ => by ring

/-- **T7.** EVPI scales by the slope. -/
theorem evpi_rescale {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K) (p : Θ → K) :
    evpi (rescale s b u) p = s * evpi u p := by
  unfold evpi
  rw [perfectValue_rescale hs, actNowValue_rescale hs]
  ring

/-- The regret table scales by the slope; the shift cancels. -/
theorem regret_rescale {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K) (a : A) (θ : Θ) :
    regret (rescale s b u) a θ = s * regret u a θ := by
  unfold regret rescale
  rw [maxOver_mul_add _ hs]
  ring

theorem maxRegret_rescale [Nonempty Θ] {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K)
    (a : A) : maxRegret (rescale s b u) a = s * maxRegret u a := by
  have h : regret (rescale s b u) a = fun θ => s * regret u a θ :=
    funext (regret_rescale hs b u a)
  unfold maxRegret
  rw [h, maxOver_mul_left _ hs]

/-- A positive affine rescaling preserves the minimax-regret action set. -/
theorem isMinimaxRegret_rescale [Nonempty Θ] {s : K} (hs : 0 < s) (b : Θ → K)
    (u : A → Θ → K) (a : A) : IsMinimaxRegret (rescale s b u) a ↔ IsMinimaxRegret u a := by
  unfold IsMinimaxRegret
  simp only [maxRegret_rescale hs.le]
  constructor
  · intro h a'
    exact le_of_mul_le_mul_left (h a') hs
  · intro h a'
    exact mul_le_mul_of_nonneg_left (h a') hs.le

theorem signalValue_rescale (s : K) (b : Θ → K) (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K)
    (z : Z) (a : A) :
    signalValue (rescale s b u) p e z a =
      s * signalValue u p e z a + ∑ θ, p θ * e θ z * b θ := by
  unfold signalValue rescale
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun θ _ => by ring

theorem valueWithInformation_rescale {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K)
    (p : Θ → K) {e : Θ → Z → K} (he : NormalisedKernel e) :
    valueWithInformation (rescale s b u) p e =
      s * valueWithInformation u p e + ∑ θ, p θ * b θ := by
  have hz : ∀ z, maxOver (signalValue (rescale s b u) p e z) =
      s * maxOver (signalValue u p e z) + ∑ θ, p θ * e θ z * b θ := by
    intro z
    have h : signalValue (rescale s b u) p e z =
        fun a => s * signalValue u p e z a + ∑ θ, p θ * e θ z * b θ :=
      funext (signalValue_rescale s b u p e z)
    rw [h, maxOver_mul_add _ hs]
  unfold valueWithInformation
  rw [Finset.sum_congr rfl fun z _ => hz z, Finset.sum_add_distrib, ← Finset.mul_sum]
  congr 1
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun θ _ => ?_
  rw [← Finset.sum_mul, ← Finset.mul_sum, he θ, mul_one]

/-- **T7.** EVSI scales by the slope. -/
theorem evsi_rescale {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K) (p : Θ → K)
    {e : Θ → Z → K} (he : NormalisedKernel e) :
    evsi (rescale s b u) p e = s * evsi u p e := by
  unfold evsi
  rw [valueWithInformation_rescale hs b u p he, actNowValue_rescale hs]
  ring

theorem normalisedKernel_deterministic [DecidableEq C] (f : Θ → C) :
    NormalisedKernel (deterministic f : Θ → C → K) := by
  intro θ
  simp [deterministic]

theorem nonnegKernel_deterministic [DecidableEq C] (f : Θ → C) :
    NonnegKernel (deterministic f : Θ → C → K) := by
  intro θ c
  unfold deterministic
  split_ifs
  · exact zero_le_one
  · exact le_rfl

theorem stochasticKernel_deterministic [DecidableEq C] (f : Θ → C) :
    StochasticKernel (deterministic f : Θ → C → K) :=
  ⟨nonnegKernel_deterministic f, normalisedKernel_deterministic f⟩

/-- **T7.** EVPPI scales by the slope. -/
theorem evppi_rescale [DecidableEq C] {s : K} (hs : 0 ≤ s) (b : Θ → K) (u : A → Θ → K)
    (p : Θ → K) (f : Θ → C) : evppi (rescale s b u) p f = s * evppi u p f := by
  rw [evppi_eq_evsi, evppi_eq_evsi, evsi_rescale hs b u p (normalisedKernel_deterministic f)]

end DecisionAnalysisProofs
