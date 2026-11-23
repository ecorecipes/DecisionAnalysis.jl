import DecisionAnalysisProofs.Finite.Extrema
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Fintype.Sum
import Mathlib.Algebra.BigOperators.Option
import Mathlib.Algebra.BigOperators.Field

/-!
# DecisionAnalysisProofs.Finite.Farkas

The one piece of linear-programming duality the package's "no" certificates rest on, proved
from first principles over any linearly ordered field (so over `ℚ` as well as `ℝ`), with no
topology and no dependency outside Mathlib.

It is here for one purpose: `Finite/Alternative.lean` uses
**`exists_mixed_pos_or_exists_mixed_nonpos`** to show that every "not E-admissible" answer of
`decide(problem, EAdmissible())` and every "not a garbling" answer of `is_garbling_of` has a
witness of the kind checked by `Certificate/Refutation.lean`. Nothing else in the project uses
this file, and the soundness of every checker is independent of it.

## Results

* **`farkas_cone`** (Farkas' lemma, cone form): for finitely many vectors `a i` and a vector
  `b`, either `b` is a nonnegative combination of the `a i`, or some `x` has
  `a i · x ≥ 0` for every `i` and `b · x < 0`. The proof is an elementary induction on the
  number of generators: if the separating `x` found for the first generators fails for the
  new one, all vectors are projected along the new one and the induction hypothesis is
  applied again.
* **`exists_mixed_pos_or_exists_mixed_nonpos`** (Ville's alternative for a matrix game):
  either the row player has a mixed strategy with positive payoff against every column, or
  the column player has a mixed strategy against which no row has positive payoff.

Not claimed: strong duality or attainment for a general linear programme, the minimax
theorem with a value, or anything about the simplex method of `src/lp.jl`.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {ι R C : Type} [Fintype ι] {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The dot product of two vectors indexed by a finite type. -/
def dot (v x : ι → K) : K := ∑ k, v k * x k

theorem dot_comb_right (v x y : ι → K) (p q : K) :
    dot v (fun k => p * x k - q * y k) = p * dot v x - q * dot v y := by
  simp only [dot, mul_sub, Finset.sum_sub_distrib, Finset.mul_sum]
  congr 1 <;> exact Finset.sum_congr rfl fun k _ => by ring

theorem dot_comb_left (v w y : ι → K) (p q : K) :
    dot (fun k => p * v k - q * w k) y = p * dot v y - q * dot w y := by
  simp only [dot, sub_mul, Finset.sum_sub_distrib, Finset.mul_sum]
  congr 1 <;> exact Finset.sum_congr rfl fun k _ => by ring

/-- **Farkas' lemma**, cone form, over an ordered field: `b` is a nonnegative combination of
the vectors `a i`, `i ∈ s`, or a vector `x` separates it from them. -/
theorem farkas_cone (s : Finset R) :
    ∀ (a : R → ι → K) (b : ι → K),
      (∃ c : R → K, (∀ i ∈ s, 0 ≤ c i) ∧ ∀ k, b k = ∑ i ∈ s, c i * a i k) ∨
      (∃ x : ι → K, (∀ i ∈ s, 0 ≤ dot (a i) x) ∧ dot b x < 0) := by
  classical
  induction s using Finset.induction_on with
  | empty =>
    intro a b
    by_cases hb : ∀ k, b k = 0
    · exact Or.inl ⟨fun _ => 0, by simp, fun k => by simp [hb k]⟩
    · refine Or.inr ⟨fun k => -b k, by simp, ?_⟩
      have hpos : 0 < ∑ k, b k * b k := by
        refine lt_of_le_of_ne (Finset.sum_nonneg fun k _ => mul_self_nonneg (b k)) fun h0 => ?_
        refine hb fun k => ?_
        exact mul_self_eq_zero.mp ((Finset.sum_eq_zero_iff_of_nonneg
          fun k _ => mul_self_nonneg (b k)).mp h0.symm k (Finset.mem_univ k))
      simp only [dot, mul_neg, Finset.sum_neg_distrib]
      linarith
  | insert n s hn ih =>
    intro a b
    rcases ih a b with ⟨c, hc, hb⟩ | ⟨x, hx, hbx⟩
    · -- `b` is already in the smaller cone.
      refine Or.inl ⟨Function.update c n 0, fun i hi => ?_, fun k => ?_⟩
      · by_cases h : i = n
        · subst h
          simp
        · rw [Function.update_of_ne h]
          exact hc i ((Finset.mem_insert.mp hi).resolve_left h)
      · rw [Finset.sum_insert hn, Function.update_self, zero_mul, zero_add, hb k]
        exact Finset.sum_congr rfl fun i hi => by
          rw [Function.update_of_ne (ne_of_mem_of_not_mem hi hn)]
    · by_cases hn0 : 0 ≤ dot (a n) x
      · -- The separating vector also works for the new generator.
        refine Or.inr ⟨x, fun i hi => ?_, hbx⟩
        rcases Finset.mem_insert.mp hi with rfl | hi
        · exact hn0
        · exact hx i hi
      · -- Project every vector along the new generator and apply the hypothesis again.
        have hα : dot (a n) x < 0 := not_le.mp hn0
        rcases ih (fun i k => dot (a i) x * a n k - dot (a n) x * a i k)
            (fun k => dot b x * a n k - dot (a n) x * b k) with ⟨c, hc, hb'⟩ | ⟨y, hy, hby⟩
        · obtain ⟨γ, hγ⟩ : ∃ γ : K,
              γ = (∑ i ∈ s, c i * dot (a i) x - dot b x) / (-dot (a n) x) := ⟨_, rfl⟩
          have hγ0 : 0 ≤ γ := by
            rw [hγ]
            have := Finset.sum_nonneg fun i hi => mul_nonneg (hc i hi) (hx i hi)
            exact div_nonneg (by linarith) (by linarith)
          have hαγ : dot (a n) x * γ = dot b x - ∑ i ∈ s, c i * dot (a i) x := by
            have h : γ * (-dot (a n) x) = ∑ i ∈ s, c i * dot (a i) x - dot b x := by
              rw [hγ]
              exact div_mul_cancel₀ _ (neg_ne_zero.mpr hα.ne)
            linarith
          refine Or.inl ⟨Function.update c n γ, fun i hi => ?_, fun k => ?_⟩
          · by_cases h : i = n
            · subst h
              rw [Function.update_self]
              exact hγ0
            · rw [Function.update_of_ne h]
              exact hc i ((Finset.mem_insert.mp hi).resolve_left h)
          · rw [Finset.sum_insert hn, Function.update_self]
            have e : ∑ i ∈ s, Function.update c n γ i * a i k = ∑ i ∈ s, c i * a i k :=
              Finset.sum_congr rfl fun i hi => by
                rw [Function.update_of_ne (ne_of_mem_of_not_mem hi hn)]
            rw [e]
            have h1 := hb' k
            have e2 : ∑ i ∈ s, c i * (dot (a i) x * a n k - dot (a n) x * a i k) =
                (∑ i ∈ s, c i * dot (a i) x) * a n k - dot (a n) x * ∑ i ∈ s, c i * a i k := by
              rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
              exact Finset.sum_congr rfl fun i _ => by ring
            rw [e2] at h1
            apply mul_left_cancel₀ hα.ne
            calc dot (a n) x * b k
                = (dot b x - ∑ i ∈ s, c i * dot (a i) x) * a n k +
                    dot (a n) x * ∑ i ∈ s, c i * a i k := by linarith
              _ = dot (a n) x * (γ * a n k + ∑ i ∈ s, c i * a i k) := by
                  rw [← hαγ]
                  ring
        · refine Or.inr ⟨fun k => dot (a n) y * x k - dot (a n) x * y k, fun i hi => ?_, ?_⟩
          · rw [dot_comb_right]
            rcases Finset.mem_insert.mp hi with rfl | hi
            · exact le_of_eq (by ring)
            · have := hy i hi
              rw [dot_comb_left] at this
              linarith
          · rw [dot_comb_right]
            have := hby
            rw [dot_comb_left] at this
            linarith

/-- **Ville's alternative.** For a matrix game `M`, either the row player has a mixed strategy
with positive payoff against every column, or the column player has a mixed strategy against
which no row has positive payoff. -/
theorem exists_mixed_pos_or_exists_mixed_nonpos [Fintype R] [Fintype C] [Nonempty C]
    (M : R → C → K) :
    (∃ x : R → K, (∀ i, 0 ≤ x i) ∧ ∑ i, x i = 1 ∧ ∀ j, 0 < ∑ i, x i * M i j) ∨
    (∃ y : C → K, (∀ j, 0 ≤ y j) ∧ ∑ j, y j = 1 ∧ ∀ i, ∑ j, M i j * y j ≤ 0) := by
  classical
  -- Coordinates: one per row and one (`none`) for the normalisation. Generators: one per
  -- column (the column of `M`, with 1 in the extra coordinate) and one slack per row.
  let gen : C ⊕ R → Option R → K := fun t o =>
    match t, o with
    | Sum.inl _, none => 1
    | Sum.inl j, some i => M i j
    | Sum.inr _, none => 0
    | Sum.inr i, some i' => if i' = i then 1 else 0
  let b : Option R → K := fun o => match o with
    | none => 1
    | some _ => 0
  rcases farkas_cone (Finset.univ : Finset (C ⊕ R)) gen b with ⟨c, hc, hb⟩ | ⟨x, hx, hbx⟩
  · refine Or.inr ⟨fun j => c (Sum.inl j), fun j => hc _ (Finset.mem_univ _), ?_, fun i => ?_⟩
    · have h := hb none
      simp only [b, gen, Fintype.sum_sum_type, mul_one, mul_zero, Finset.sum_const_zero,
        add_zero] at h
      exact h.symm
    · have h := hb (some i)
      simp only [b, gen, Fintype.sum_sum_type, mul_ite, mul_one, mul_zero,
        Finset.sum_ite_eq, Finset.mem_univ, if_true] at h
      have h0 := hc (Sum.inr i) (Finset.mem_univ _)
      have e : ∑ j, M i j * c (Sum.inl j) = ∑ j, c (Sum.inl j) * M i j :=
        Finset.sum_congr rfl fun j _ => mul_comm _ _
      rw [e]
      linarith
  · have hnone : x none < 0 := by
      simpa [dot, b, Fintype.sum_option] using hbx
    have hcol : ∀ j, 0 < ∑ i, x (some i) * M i j := fun j => by
      have h := hx (Sum.inl j) (Finset.mem_univ _)
      simp only [dot, gen, Fintype.sum_option, one_mul] at h
      have e : ∑ i, M i j * x (some i) = ∑ i, x (some i) * M i j :=
        Finset.sum_congr rfl fun i _ => mul_comm _ _
      rw [e] at h
      linarith
    have hrow : ∀ i, 0 ≤ x (some i) := fun i => by
      have h := hx (Sum.inr i) (Finset.mem_univ _)
      simpa [dot, gen, Fintype.sum_option] using h
    have hS : 0 < ∑ i, x (some i) := by
      refine lt_of_le_of_ne (Finset.sum_nonneg fun i _ => hrow i) fun h0 => ?_
      obtain ⟨j⟩ := ‹Nonempty C›
      have hz := (Finset.sum_eq_zero_iff_of_nonneg fun i _ => hrow i).mp h0.symm
      have := hcol j
      rw [Finset.sum_eq_zero fun i hi => by rw [hz i hi, zero_mul]] at this
      exact lt_irrefl _ this
    refine Or.inl ⟨fun i => x (some i) / ∑ i, x (some i), fun i => div_nonneg (hrow i) hS.le,
      ?_, fun j => ?_⟩
    · rw [← Finset.sum_div, div_self hS.ne']
    · have e : ∑ i, x (some i) / (∑ i, x (some i)) * M i j =
          (∑ i, x (some i) * M i j) / ∑ i, x (some i) := by
        rw [Finset.sum_div]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [e]
      exact div_pos (hcol j) hS

end DecisionAnalysisProofs
