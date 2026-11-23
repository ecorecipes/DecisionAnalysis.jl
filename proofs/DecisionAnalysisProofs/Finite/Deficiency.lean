import DecisionAnalysisProofs.Finite.Fidelity
import DecisionAnalysisProofs.Finite.Garbling

/-!
# DecisionAnalysisProofs.Finite.Deficiency

Le Cam's deficiency and the value gap it bounds (`deficiency`, `src/experiments.jl`; programme
`deficiency_model`, `src/programs.jl`).

## Conventions

* Julia computes `δ(e, e′) = min_G max_θ TV(G · L_e[·, θ], L_e′[·, θ])` over column-stochastic
  `G`, with `TV` half the ℓ¹ distance. Lean does not model the minimisation. It states what a
  feasible point of the programme is: `IsDeficientWithin e e' δ` says that some stochastic `g`
  brings `garble e g` within total variation `δ` of `e'` in every state. The Julia value
  `deficiency(e, e′)` is the least such `δ`, so `IsDeficientWithin e e' (deficiency(e, e′))`
  holds, and every theorem below applies to it.
* The payoff range enters as `∀ θ a a', u a θ - u a' θ ≤ B`: within each **column** (state) of
  the payoff table, payoffs differ by at most `B`. The range of the whole table satisfies it.

## What each result backs

* **T6a**, **`IsDeficientWithin.valueWithInformation_le`** and **`IsDeficientWithin.evsi_le`**
  back the docstring of `deficiency`: "the value of acting on `e′` exceeds that of acting on
  `e` by at most `B δ(e, e′)`, for every prior". The constant is `B`, not `2 B`: the
  comparison is between two strategies evaluated under the *same* prior, so only one
  total-variation step is paid (unlike `decision_loss_bound`, which pays two).
* **`isDeficientWithin_zero_iff`** backs "`δ(e, e′) = 0` exactly when `e′` is a garbling of
  `e`" in the same docstring, and the assertion
  `is_garbling == iszero(deficiency(e, f))` of `test/test_programs.jl` and
  `test/test_experiments.jl`.
* **`IsDeficientWithin.mono`**, **`IsGarblingOf.isDeficientWithin`**: larger tolerances and
  garblings.

The hypothesis on `B` is weaker than the Julia docstring's (table range): the bound holds with
`B` the largest **within-state** payoff spread `max_θ (max_a Q[a, θ] − min_a Q[a, θ])`, which
is the largest entry of the regret table (`maximum(regret_table(problem))`). See
`Roadmap.lean`.

Not claimed: that the linear programme attains the minimum, the converse bound (S1a in
`Roadmap.lean`), or anything for a prior that is not a probability vector beyond
`valueWithInformation_le_add_of_totalVariation_le`, which carries the total mass explicitly.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z Z' : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z] [Fintype Z']
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- A feasible point of the deficiency programme with objective `δ`: some garbling of `e` is
within total variation `δ` of `e'` in every state. `deficiency(e, e′)` is the least such `δ`. -/
def IsDeficientWithin (e : Θ → Z → K) (e' : Θ → Z' → K) (δ : K) : Prop :=
  ∃ g : Z → Z' → K, StochasticKernel g ∧ ∀ θ, totalVariation (garble e g θ) (e' θ) ≤ δ

omit [Fintype A] [Nonempty A] in
theorem IsDeficientWithin.mono {e : Θ → Z → K} {e' : Θ → Z' → K} {δ δ' : K}
    (h : IsDeficientWithin e e' δ) (hδ : δ ≤ δ') : IsDeficientWithin e e' δ' := by
  obtain ⟨g, hg, hd⟩ := h
  exact ⟨g, hg, fun θ => (hd θ).trans hδ⟩

omit [Fintype A] [Nonempty A] in
/-- Deficiency zero is garbling (`is_garbling == iszero(deficiency(e, e′))`). -/
theorem isDeficientWithin_zero_iff (e : Θ → Z → K) (e' : Θ → Z' → K) :
    IsDeficientWithin e e' 0 ↔ IsGarblingOf e' e := by
  constructor
  · rintro ⟨g, hg, hd⟩
    refine ⟨g, hg, funext fun θ => ?_⟩
    exact ((totalVariation_eq_zero_iff _ _).mp
      (le_antisymm (hd θ) (totalVariation_nonneg _ _))).symm
  · rintro ⟨g, hg, rfl⟩
    exact ⟨g, hg, fun θ => (totalVariation_self _).le⟩

omit [Fintype A] [Nonempty A] in
/-- A garbling is within every nonnegative tolerance. -/
theorem IsGarblingOf.isDeficientWithin {e' : Θ → Z' → K} {e : Θ → Z → K}
    (h : IsGarblingOf e' e) {δ : K} (hδ : 0 ≤ δ) : IsDeficientWithin e e' δ :=
  ((isDeficientWithin_zero_iff e e').mpr h).mono hδ

/-- Two experiments on the same signals whose rows have equal mass and are within total
variation `δ` in every state have values within `B · δ · Σ p`, for a nonnegative `p`. -/
theorem valueWithInformation_le_add_of_totalVariation_le (u : A → Θ → K) {p : Θ → K}
    (hp : NonnegPrior p) {e₁ e₂ : Θ → Z' → K} (hmass : ∀ θ, ∑ z, e₂ θ z = ∑ z, e₁ θ z)
    {B δ : K} (hB : ∀ θ a a', u a θ - u a' θ ≤ B)
    (hδ : ∀ θ, totalVariation (e₁ θ) (e₂ θ) ≤ δ) :
    valueWithInformation u p e₂ ≤ valueWithInformation u p e₁ + B * δ * ∑ θ, p θ := by
  -- The best response to each signal of `e₂`.
  choose d hd using fun z => exists_eq_maxOver (signalValue u p e₂ z)
  have h₂ : valueWithInformation u p e₂ = ∑ θ, p θ * ∑ z, e₂ θ z * u (d z) θ := by
    unfold valueWithInformation
    rw [Finset.sum_congr rfl fun z _ => hd z]
    unfold signalValue
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun θ _ => by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun z _ => by ring
  have h₁ : ∑ θ, p θ * ∑ z, e₁ θ z * u (d z) θ ≤ valueWithInformation u p e₁ := by
    have : ∑ θ, p θ * ∑ z, e₁ θ z * u (d z) θ = ∑ z, signalValue u p e₁ z (d z) := by
      unfold signalValue
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun θ _ => by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun z _ => by ring
    rw [this]
    exact Finset.sum_le_sum fun z _ => le_maxOver (signalValue u p e₁ z) (d z)
  have hstate : ∀ θ, p θ * ∑ z, e₂ θ z * u (d z) θ ≤
      p θ * ∑ z, e₁ θ z * u (d z) θ + B * δ * p θ := by
    intro θ
    have hB0 : 0 ≤ B := by
      obtain ⟨a⟩ := ‹Nonempty A›
      simpa using hB θ a a
    have h := sum_sub_mul_le (e₂ θ) (e₁ θ) (hmass θ) (fun z => u (d z) θ)
      fun z z' => hB θ (d z) (d z')
    rw [totalVariation_comm] at h
    have h' : ∑ z, e₂ θ z * u (d z) θ - ∑ z, e₁ θ z * u (d z) θ ≤ B * δ := by
      rw [← Finset.sum_sub_distrib]
      calc ∑ z, (e₂ θ z * u (d z) θ - e₁ θ z * u (d z) θ)
          = ∑ z, (e₂ θ z - e₁ θ z) * u (d z) θ :=
            Finset.sum_congr rfl fun z _ => by ring
        _ ≤ B * totalVariation (e₁ θ) (e₂ θ) := h
        _ ≤ B * δ := mul_le_mul_of_nonneg_left (hδ θ) hB0
    have := mul_le_mul_of_nonneg_left h' (hp θ)
    linarith
  calc valueWithInformation u p e₂
      = ∑ θ, p θ * ∑ z, e₂ θ z * u (d z) θ := h₂
    _ ≤ ∑ θ, (p θ * ∑ z, e₁ θ z * u (d z) θ + B * δ * p θ) :=
        Finset.sum_le_sum fun θ _ => hstate θ
    _ = ∑ θ, p θ * ∑ z, e₁ θ z * u (d z) θ + B * δ * ∑ θ, p θ := by
        rw [Finset.sum_add_distrib, Finset.mul_sum]
    _ ≤ valueWithInformation u p e₁ + B * δ * ∑ θ, p θ := by linarith

/-- **T6a** (`deficiency`). If `e` simulates `e'` to within deficiency `δ`, then for every
probability prior the value of acting on `e'` exceeds that of acting on `e` by at most
`B · δ`, with `B` the largest spread of payoffs within a state. -/
theorem IsDeficientWithin.valueWithInformation_le {e : Θ → Z → K} {e' : Θ → Z' → K} {δ : K}
    (h : IsDeficientWithin e e' δ) (he : NormalisedKernel e) (he' : NormalisedKernel e')
    (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p) (hp1 : NormalisedPrior p) {B : K}
    (hB : ∀ θ a a', u a θ - u a' θ ≤ B) :
    valueWithInformation u p e' ≤ valueWithInformation u p e + B * δ := by
  obtain ⟨g, hg, hd⟩ := h
  have hmass : ∀ θ, ∑ z, e' θ z = ∑ z, garble e g θ z := fun θ => by
    rw [he' θ, he.garble hg.2 θ]
  have h1 := valueWithInformation_le_add_of_totalVariation_le u hp hmass hB hd
  have h2 := valueWithInformation_garble_le u p e hg
  unfold NormalisedPrior at hp1
  rw [hp1, mul_one] at h1
  linarith

/-- **T6a**, for EVSI. -/
theorem IsDeficientWithin.evsi_le {e : Θ → Z → K} {e' : Θ → Z' → K} {δ : K}
    (h : IsDeficientWithin e e' δ) (he : NormalisedKernel e) (he' : NormalisedKernel e')
    (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p) (hp1 : NormalisedPrior p) {B : K}
    (hB : ∀ θ a a', u a θ - u a' θ ≤ B) :
    evsi u p e' ≤ evsi u p e + B * δ := by
  have := h.valueWithInformation_le he he' u hp hp1 hB
  unfold evsi
  linarith

end DecisionAnalysisProofs
