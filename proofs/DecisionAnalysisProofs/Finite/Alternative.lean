import DecisionAnalysisProofs.Finite.Credal
import DecisionAnalysisProofs.Finite.Garbling
import DecisionAnalysisProofs.Finite.Farkas
import Mathlib.Tactic.FieldSimp

/-!
# DecisionAnalysisProofs.Finite.Alternative

Completeness of the two "no" certificates of `Certificate/Refutation.lean`: whenever the
answer of `decide(problem, EAdmissible())` or of `is_garbling_of` is "no", a witness of the
exported kind exists, so a checked certificate can accompany **every** answer, positive or
negative, over any ordered field and in particular over `ℚ`.

## Route

The only non-elementary step is Ville's alternative for a matrix game,
`exists_mixed_pos_or_exists_mixed_nonpos` of `Finite/Farkas.lean`, proved there from Farkas'
lemma over any linearly ordered field. No external library is used.

## What each result backs

* **T12b**, **`not_isEAdmissible_iff_isMixedDominated`**: an action is not E-admissible iff
  some mixed action strictly dominates it at every generating point of the credal set.
  Backs the completeness of `"not-e-admissible-witness/1"`: every action left out by
  `decide(problem, EAdmissible())` has a certificate. It is also the statement that the
  optimum of `eadmissibility_model` is negative exactly when its dual has a strictly
  dominating mixed action, which is where the Julia exporter finds the witness.
* **S1** (Blackwell's theorem, the converse direction),
  **`isGarblingOf_of_forall_valueWithInformation_le`**: if `e′` is worth no more than `e` for
  every payoff table whose actions are the signals of `e′`, under **one** prior of full
  support, then `e′` is a garbling of `e`. **`not_isGarblingOf_iff_exists_separating`**:
  `e′` is not a garbling of `e` iff some such payoff table gives `e′` strictly higher value.
  Backs the completeness of `"not-garbling-witness/1"`: every `nothing` returned by
  `is_garbling_of` has a certificate, and that certificate may always use the uniform prior
  and one action per signal of `e′`. **`isGarblingOf_iff_forall_valueWithInformation_le`** is
  the two-sided statement behind the docstring of `garble` ("less informative in Blackwell's
  order").

Edge cases: the prior must have full support (a state of prior probability zero is invisible
to every decision problem, so no problem with that prior can separate experiments that differ
only there); `e′` must have at least one signal. Neither experiment need be stochastic.

Not claimed: the approximate version (S1a: value gap at most `B δ` for all problems iff
deficiency at most `δ`), and nothing about how Julia extracts the witness from its solver.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ I Z Z' : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype I] [Nonempty I]
  [Fintype Z] [Fintype Z'] {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### T12b: E-admissibility -/

/-- **T12b.** Every action is E-admissible or strictly dominated by a mixed action at every
generating point. -/
theorem isEAdmissible_or_isMixedDominated (u : A → Θ → K) (P : I → Θ → K) (a : A) :
    IsEAdmissible u P a ∨ IsMixedDominated u P a := by
  rcases exists_mixed_pos_or_exists_mixed_nonpos
    (fun (b : A) (i : I) => expectedUtility u (P i) b - expectedUtility u (P i) a) with
    ⟨m, hm, hm1, hpos⟩ | ⟨w, hw, hw1, hle⟩
  · refine Or.inr ⟨m, hm, hm1, fun i => ?_⟩
    have h := hpos i
    simp only [mul_sub, Finset.sum_sub_distrib] at h
    rw [← Finset.sum_mul, hm1, one_mul] at h
    linarith
  · refine Or.inl ⟨w, hw, hw1, fun b => ?_⟩
    have h := hle b
    simp only [sub_mul, Finset.sum_sub_distrib] at h
    rw [expectedUtility_mixture, expectedUtility_mixture]
    have e : ∀ c : A, ∑ i, w i * expectedUtility u (P i) c =
        ∑ i, expectedUtility u (P i) c * w i := fun c =>
      Finset.sum_congr rfl fun i _ => mul_comm _ _
    rw [e, e]
    linarith

/-- **T12b.** An action is not E-admissible exactly when a mixed action strictly dominates it
at every generating point of the credal set. -/
theorem not_isEAdmissible_iff_isMixedDominated (u : A → Θ → K) (P : I → Θ → K) (a : A) :
    ¬ IsEAdmissible u P a ↔ IsMixedDominated u P a :=
  ⟨fun h => (isEAdmissible_or_isMixedDominated u P a).resolve_left h,
    fun h => h.not_isEAdmissible⟩

/-! ### S1: Blackwell's theorem, the converse direction -/

/-- The sign `+1` or `-1`. -/
def signOf (b : Bool) : K := if b then 1 else -1

omit [Fintype A] [Nonempty A] [Fintype I] [Nonempty I] in
/-- A mixture of deterministic garblings is a garbling by the mixed kernel. -/
theorem sum_mul_garble_deterministic [DecidableEq Z] [DecidableEq Z'] (e : Θ → Z → K)
    (x : (Z → Z') → K) (θ : Θ) (z' : Z') :
    ∑ f, x f * garble e (deterministic f) θ z' =
      garble e (fun z z' => ∑ f, x f * deterministic f z z') θ z' := by
  simp only [garble, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun z _ => Finset.sum_congr rfl fun f _ => by ring

omit [Fintype A] [Nonempty A] [Fintype I] [Nonempty I] in
/-- **S1** (Blackwell). If, under one prior of full support, `e'` is worth no more than `e`
for every payoff table whose actions are the signals of `e'`, then `e'` is a garbling of
`e`. -/
theorem isGarblingOf_of_forall_valueWithInformation_le [Nonempty Z'] {e : Θ → Z → K}
    {e' : Θ → Z' → K} {p : Θ → K} (hp : ∀ θ, 0 < p θ)
    (h : ∀ u : Z' → Θ → K, valueWithInformation u p e' ≤ valueWithInformation u p e) :
    IsGarblingOf e' e := by
  classical
  -- The gap between `e'` and the deterministic garbling by `f`, at the pair `r = (θ, z')`.
  let D : (Z → Z') → Θ × Z' → K := fun f r => e' r.1 r.2 - garble e (deterministic f) r.1 r.2
  rcases exists_mixed_pos_or_exists_mixed_nonpos
    (fun (s : Θ × Z' → Bool) (f : Z → Z') => ∑ r, signOf (s r) * D f r) with
    ⟨y, hy, hy1, hpos⟩ | ⟨x, hx, hx1, hle⟩
  · -- A payoff table on which `e'` strictly beats `e`: contradiction.
    exfalso
    let T : Θ × Z' → K := fun r => ∑ s, y s * signOf (s r)
    have hT : ∀ f, 0 < ∑ r, T r * D f r := fun f => by
      have h1 := hpos f
      have e1 : ∑ s, y s * ∑ r, signOf (s r) * D f r = ∑ r, T r * D f r := by
        simp only [T, Finset.mul_sum, Finset.sum_mul]
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun s _ => by ring
      rwa [e1] at h1
    let u : Z' → Θ → K := fun a θ => T (θ, a) / p θ
    have hsv : ∀ {Y : Type} [Fintype Y] (e₀ : Θ → Y → K) (z : Y) (a : Z'),
        signalValue u p e₀ z a = ∑ θ, e₀ θ z * T (θ, a) := fun e₀ z a =>
      Finset.sum_congr rfl fun θ _ => by
        have := (hp θ).ne'
        simp only [u]
        field_simp
    choose f hf using fun z => exists_eq_maxOver (signalValue u p e z)
    have hval : valueWithInformation u p e = ∑ r, T r * garble e (deterministic f) r.1 r.2 := by
      unfold valueWithInformation
      rw [Finset.sum_congr rfl fun z _ => hf z, Fintype.sum_prod_type]
      simp only [hsv, garble, Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun θ _ => ?_
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun z _ => ?_
      simp [deterministic, mul_comm]
    have hval' : ∑ r, T r * e' r.1 r.2 ≤ valueWithInformation u p e' := by
      have : ∑ r, T r * e' r.1 r.2 = ∑ z', signalValue u p e' z' z' := by
        rw [Fintype.sum_prod_type, Finset.sum_comm]
        simp only [hsv]
        exact Finset.sum_congr rfl fun z' _ => Finset.sum_congr rfl fun θ _ => mul_comm _ _
      rw [this]
      exact Finset.sum_le_sum fun z' _ => le_maxOver (signalValue u p e' z') z'
    have h2 := hT f
    simp only [D, mul_sub, Finset.sum_sub_distrib] at h2
    have h3 := h u
    linarith
  · -- A mixture of deterministic garblings reproduces `e'`.
    let g : Z → Z' → K := fun z z' => ∑ f, x f * deterministic f z z'
    have hg : StochasticKernel g := by
      refine ⟨fun z z' => Finset.sum_nonneg fun f _ => mul_nonneg (hx f) ?_, fun z => ?_⟩
      · exact (stochasticKernel_deterministic f).1 z z'
      · simp only [g]
        rw [Finset.sum_comm]
        calc ∑ f, ∑ z', x f * deterministic f z z'
            = ∑ f, x f := Finset.sum_congr rfl fun f _ => by
              rw [← Finset.mul_sum, (stochasticKernel_deterministic (K := K) f).2 z, mul_one]
          _ = 1 := hx1
    have hmix : ∀ s : Θ × Z' → Bool,
        ∑ r, signOf (s r) * (e' r.1 r.2 - garble e g r.1 r.2) ≤ 0 := fun s => by
      have h1 := hle s
      have e1 : ∑ f, (∑ r, signOf (s r) * D f r) * x f =
          ∑ r, signOf (s r) * (e' r.1 r.2 - garble e g r.1 r.2) := by
        simp only [Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun r _ => ?_
        have e2 : ∑ f, x f * D f r = e' r.1 r.2 - garble e g r.1 r.2 := by
          simp only [D, mul_sub, Finset.sum_sub_distrib]
          rw [← Finset.sum_mul, hx1, one_mul, sum_mul_garble_deterministic]
        rw [← e2, Finset.mul_sum]
        exact Finset.sum_congr rfl fun f _ => by ring
      rwa [e1] at h1
    have habs : ∑ r : Θ × Z', |e' r.1 r.2 - garble e g r.1 r.2| ≤ 0 := by
      have h1 := hmix fun r => decide (0 ≤ e' r.1 r.2 - garble e g r.1 r.2)
      refine le_trans (le_of_eq (Finset.sum_congr rfl fun r _ => ?_)) h1
      by_cases hr : 0 ≤ e' r.1 r.2 - garble e g r.1 r.2
      · simp [signOf, hr, abs_of_nonneg hr]
      · simp [signOf, hr, abs_of_neg (not_le.mp hr)]
    have hzero := (Finset.sum_eq_zero_iff_of_nonneg
      fun r _ => abs_nonneg (e' r.1 r.2 - garble e g r.1 r.2)).mp
        (le_antisymm habs (Finset.sum_nonneg fun r _ => abs_nonneg _))
    refine ⟨g, hg, funext fun θ => funext fun z' => ?_⟩
    exact sub_eq_zero.mp (abs_eq_zero.mp (hzero (θ, z') (Finset.mem_univ _)))

omit [Fintype A] [Nonempty A] [Fintype I] [Nonempty I] in
/-- **S1**, as a certificate statement: `e'` is not a garbling of `e` exactly when some payoff
table, with the signals of `e'` as actions, gives `e'` strictly higher value under the given
prior of full support. -/
theorem not_isGarblingOf_iff_exists_separating [Nonempty Z'] (e : Θ → Z → K)
    (e' : Θ → Z' → K) {p : Θ → K} (hp : ∀ θ, 0 < p θ) :
    ¬ IsGarblingOf e' e ↔
      ∃ u : Z' → Θ → K, valueWithInformation u p e < valueWithInformation u p e' := by
  constructor
  · intro hng
    by_contra hno
    exact hng (isGarblingOf_of_forall_valueWithInformation_le hp fun u =>
      not_lt.mp fun hu => hno ⟨u, hu⟩)
  · rintro ⟨u, hu⟩ hg
    exact absurd (hg.valueWithInformation_le u p) (not_le.mpr hu)

omit [Fintype A] [Nonempty A] [Fintype I] [Nonempty I] in
/-- **Blackwell's theorem** on finite spaces: `e'` is a garbling of `e` iff it is worth no more
than `e` in every finite decision problem with a probability prior. -/
theorem isGarblingOf_iff_forall_valueWithInformation_le [Nonempty Θ] [Nonempty Z']
    (e : Θ → Z → K) (e' : Θ → Z' → K) :
    IsGarblingOf e' e ↔
      ∀ (B : Type) [Fintype B] [Nonempty B] (u : B → Θ → K) (p : Θ → K),
        NonnegPrior p → NormalisedPrior p →
          valueWithInformation u p e' ≤ valueWithInformation u p e := by
  constructor
  · intro hg B _ _ u p _ _
    exact hg.valueWithInformation_le u p
  · intro h
    have hcard : (0 : K) < Fintype.card Θ := by exact_mod_cast Fintype.card_pos
    refine isGarblingOf_of_forall_valueWithInformation_le
      (p := fun _ => 1 / (Fintype.card Θ : K)) (fun _ => one_div_pos.mpr hcard) fun u => ?_
    refine h Z' u _ (fun _ => (one_div_pos.mpr hcard).le) ?_
    unfold NormalisedPrior
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    field_simp

end DecisionAnalysisProofs
