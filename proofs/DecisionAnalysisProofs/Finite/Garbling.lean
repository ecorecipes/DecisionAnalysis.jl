import DecisionAnalysisProofs.Finite.Affine

/-!
# DecisionAnalysisProofs.Finite.Garbling

Garbling of experiments and the easy direction of Blackwell's theorem (`src/experiments.jl`:
`garble`, `is_garbling_of`; `src/voi.jl`: `value_with_information`, `evsi`).

## Conventions

* `garble e g` is `e` followed by `g`: `(garble e g) θ z' = Σ_z e θ z · g z z'`. At `K = ℝ` it
  is `FiniteKernelsProofs.Finite.Kernel.comp e g` (`Theory/FiniteKernels.lean`).
* `IsGarblingOf e' e` says that some *stochastic* `g` gives `e' = garble e g`. The two
  experiments share the state space and may have different signal spaces.

## Results

* **T0**, **`IsGarblingOf.refl`** and **`IsGarblingOf.trans`**: garbling is a preorder
  (reflexive through the identity kernel `deterministic id`, transitive through composition).
* **T6**, **`valueWithInformation_garble_le`**: `value(garble e g) ≤ value(e)` for every payoff
  table `u` and every `p`, when `g` is stochastic. Neither nonnegativity of `p` nor any
  property of `e` is used, so the theorem covers every nonnegative prior in particular.
  **`IsGarblingOf.valueWithInformation_le`** and **`IsGarblingOf.evsi_le`** restate it.

The proof is the simulation argument: the objective after a garbled signal is a nonnegative
mixture of the objectives after the original signals, and a mixture of maxima dominates the
maximum of the mixture (`maxOver_weighted_sum_le`).

Not claimed: the converse (Blackwell's theorem proper, S1 in `Roadmap.lean`), and nothing
about how `is_garbling_of` searches for `g` in Julia.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z Z' Z'' : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z] [Fintype Z']
  [Fintype Z''] {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- `garble(e, g)`: the experiment `e` followed by the kernel `g` on signals. -/
def garble (e : Θ → Z → K) (g : Z → Z' → K) : Θ → Z' → K := fun θ z' => ∑ z, e θ z * g z z'

/-- `is_garbling_of(e', e)`: `e'` is `e` followed by some stochastic kernel. -/
def IsGarblingOf (e' : Θ → Z' → K) (e : Θ → Z → K) : Prop :=
  ∃ g : Z → Z' → K, StochasticKernel g ∧ e' = garble e g

theorem garble_assoc (e : Θ → Z → K) (g : Z → Z' → K) (h : Z' → Z'' → K) :
    garble (garble e g) h = garble e (garble g h) := by
  funext θ z''
  simp only [garble, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun z _ => Finset.sum_congr rfl fun z' _ => by ring

theorem garble_identity [DecidableEq Z] (e : Θ → Z → K) : garble e (deterministic id) = e := by
  funext θ z
  simp [garble, deterministic]

theorem identity_garble [DecidableEq Θ] (e : Θ → Z → K) : garble (deterministic id) e = e := by
  funext θ z
  simp [garble, deterministic]

theorem NonnegKernel.garble {e : Θ → Z → K} {g : Z → Z' → K} (he : NonnegKernel e)
    (hg : NonnegKernel g) : NonnegKernel (garble e g) := fun θ z' =>
  Finset.sum_nonneg fun z _ => mul_nonneg (he θ z) (hg z z')

theorem NormalisedKernel.garble {e : Θ → Z → K} {g : Z → Z' → K} (he : NormalisedKernel e)
    (hg : NormalisedKernel g) : NormalisedKernel (garble e g) := by
  intro θ
  unfold DecisionAnalysisProofs.garble
  rw [Finset.sum_comm]
  rw [← he θ]
  refine Finset.sum_congr rfl fun z _ => ?_
  rw [← Finset.mul_sum, hg z, mul_one]

theorem StochasticKernel.garble {e : Θ → Z → K} {g : Z → Z' → K} (he : StochasticKernel e)
    (hg : StochasticKernel g) : StochasticKernel (garble e g) :=
  ⟨he.1.garble hg.1, he.2.garble hg.2⟩

/-- **T0.** Garbling is reflexive: every experiment is itself followed by the identity. -/
theorem IsGarblingOf.refl [DecidableEq Z] (e : Θ → Z → K) : IsGarblingOf e e :=
  ⟨deterministic id, stochasticKernel_deterministic id, (garble_identity e).symm⟩

/-- **T0.** Garbling is transitive: garblings compose. -/
theorem IsGarblingOf.trans {e'' : Θ → Z'' → K} {e' : Θ → Z' → K} {e : Θ → Z → K}
    (h₂ : IsGarblingOf e'' e') (h₁ : IsGarblingOf e' e) : IsGarblingOf e'' e := by
  obtain ⟨g₂, hg₂, rfl⟩ := h₂
  obtain ⟨g₁, hg₁, rfl⟩ := h₁
  exact ⟨garble g₁ g₂, hg₁.garble hg₂, garble_assoc e g₁ g₂⟩

/-- The objective after a garbled signal is a mixture of the objectives after the original
signals. -/
theorem signalValue_garble (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) (g : Z → Z' → K)
    (z' : Z') (a : A) :
    signalValue u p (garble e g) z' a = ∑ z, g z z' * signalValue u p e z a := by
  simp only [signalValue, garble, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun z _ => Finset.sum_congr rfl fun θ _ => by ring

/-- **T6** (Blackwell, easy direction). Garbling an experiment cannot raise the value of
deciding after it, for every payoff table and every prior. -/
theorem valueWithInformation_garble_le (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K)
    {g : Z → Z' → K} (hg : StochasticKernel g) :
    valueWithInformation u p (garble e g) ≤ valueWithInformation u p e := by
  calc valueWithInformation u p (garble e g)
      = ∑ z', maxOver fun a => ∑ z, g z z' * signalValue u p e z a := by
        unfold valueWithInformation
        exact Finset.sum_congr rfl fun z' _ =>
          congrArg maxOver (funext (signalValue_garble u p e g z'))
    _ ≤ ∑ z', ∑ z, g z z' * maxOver (signalValue u p e z) :=
        Finset.sum_le_sum fun z' _ =>
          maxOver_weighted_sum_le (fun z => g z z') (fun z => hg.1 z z') _
    _ = valueWithInformation u p e := by
        unfold valueWithInformation
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun z _ => by rw [← Finset.sum_mul, hg.2 z, one_mul]

/-- **T6**, for the garbling relation. -/
theorem IsGarblingOf.valueWithInformation_le {e' : Θ → Z' → K} {e : Θ → Z → K}
    (h : IsGarblingOf e' e) (u : A → Θ → K) (p : Θ → K) :
    valueWithInformation u p e' ≤ valueWithInformation u p e := by
  obtain ⟨g, hg, rfl⟩ := h
  exact valueWithInformation_garble_le u p e hg

/-- **T6**, for EVSI: a garbling is worth no more than the original experiment. -/
theorem IsGarblingOf.evsi_le {e' : Θ → Z' → K} {e : Θ → Z → K}
    (h : IsGarblingOf e' e) (u : A → Θ → K) (p : Θ → K) : evsi u p e' ≤ evsi u p e := by
  have := h.valueWithInformation_le u p
  unfold evsi
  linarith

end DecisionAnalysisProofs
