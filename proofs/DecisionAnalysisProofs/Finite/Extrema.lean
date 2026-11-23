import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FinCases

/-!
# DecisionAnalysisProofs.Finite.Extrema

Attained maxima and minima of a function on a finite nonempty type. Every "best action",
"worst state" and "worst prior" in `DecisionAnalysis.jl` (`maximum`/`minimum` over a row or a
column of a payoff table in `src/rules.jl`, `src/regret.jl` and `src/voi.jl`) is one of these.

## Conventions

* `maxOver f` is `Finset.sup'` over `Finset.univ`, with the nonemptiness proof supplied by the
  `Nonempty` instance; `minOver f` is the dual `Finset.inf'`. Neither is a bare supremum:
  **`exists_eq_maxOver`** and **`exists_eq_minOver`** give an index at which the value is
  attained.
* The order is any linear order `K`; the affine lemmas need an ordered field. The same
  definitions are therefore used over `ℝ` (theorems) and over `ℚ` (worked examples and the
  certificate checker), with no second set of definitions to relate.
* `maxOver_fin_two`, `maxOver_fin_three`, `maxOver_fin_four` and their `minOver` duals unfold
  the extremum on small `Fin n` into nested `max`/`min`, for evaluation by `simp`/`norm_num`.

Not claimed: anything about which maximiser Julia's `argmax` returns on ties.
-/

set_option autoImplicit false

namespace DecisionAnalysisProofs

open Finset

section Order

variable {ι K : Type} [Fintype ι] [Nonempty ι] [LinearOrder K]

/-- The attained maximum of `f` over a finite nonempty type (`maximum(f, xs)` in Julia). -/
def maxOver (f : ι → K) : K := Finset.univ.sup' Finset.univ_nonempty f

/-- The attained minimum of `f` over a finite nonempty type (`minimum(f, xs)` in Julia). -/
def minOver (f : ι → K) : K := Finset.univ.inf' Finset.univ_nonempty f

theorem le_maxOver (f : ι → K) (i : ι) : f i ≤ maxOver f :=
  Finset.le_sup' f (Finset.mem_univ i)

theorem maxOver_le {f : ι → K} {c : K} (h : ∀ i, f i ≤ c) : maxOver f ≤ c :=
  Finset.sup'_le _ _ fun i _ => h i

/-- The maximum is attained. -/
theorem exists_eq_maxOver (f : ι → K) : ∃ i, maxOver f = f i := by
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty f
  exact ⟨i, hi⟩

theorem maxOver_eq_of {f : ι → K} {c : K} (i : ι) (hi : f i = c) (h : ∀ j, f j ≤ c) :
    maxOver f = c :=
  le_antisymm (maxOver_le h) (hi ▸ le_maxOver f i)

theorem maxOver_mono {f g : ι → K} (h : ∀ i, f i ≤ g i) : maxOver f ≤ maxOver g :=
  maxOver_le fun i => (h i).trans (le_maxOver g i)

theorem minOver_le (f : ι → K) (i : ι) : minOver f ≤ f i :=
  Finset.inf'_le f (Finset.mem_univ i)

theorem le_minOver {f : ι → K} {c : K} (h : ∀ i, c ≤ f i) : c ≤ minOver f :=
  Finset.le_inf' _ _ fun i _ => h i

/-- The minimum is attained. -/
theorem exists_eq_minOver (f : ι → K) : ∃ i, minOver f = f i := by
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty f
  exact ⟨i, hi⟩

theorem minOver_eq_of {f : ι → K} {c : K} (i : ι) (hi : f i = c) (h : ∀ j, c ≤ f j) :
    minOver f = c :=
  le_antisymm (hi ▸ minOver_le f i) (le_minOver h)

theorem minOver_le_maxOver (f : ι → K) : minOver f ≤ maxOver f := by
  obtain ⟨i⟩ := ‹Nonempty ι›
  exact (minOver_le f i).trans (le_maxOver f i)

end Order

section Small

variable {K : Type} [LinearOrder K]

theorem maxOver_unit (f : Unit → K) : maxOver f = f () :=
  maxOver_eq_of () rfl fun _ => le_rfl

theorem maxOver_fin_two (f : Fin 2 → K) : maxOver f = max (f 0) (f 1) := by
  apply le_antisymm
  · apply maxOver_le
    intro i
    fin_cases i
    · exact le_max_left _ _
    · exact le_max_right _ _
  · exact max_le (le_maxOver f 0) (le_maxOver f 1)

theorem maxOver_fin_three (f : Fin 3 → K) : maxOver f = max (max (f 0) (f 1)) (f 2) := by
  apply le_antisymm
  · apply maxOver_le
    intro i
    fin_cases i
    · exact (le_max_left _ _).trans (le_max_left _ _)
    · exact (le_max_right _ _).trans (le_max_left _ _)
    · exact le_max_right _ _
  · exact max_le (max_le (le_maxOver f 0) (le_maxOver f 1)) (le_maxOver f 2)

theorem maxOver_fin_four (f : Fin 4 → K) :
    maxOver f = max (max (max (f 0) (f 1)) (f 2)) (f 3) := by
  apply le_antisymm
  · apply maxOver_le
    intro i
    fin_cases i
    · exact ((le_max_left _ _).trans (le_max_left _ _)).trans (le_max_left _ _)
    · exact ((le_max_right _ _).trans (le_max_left _ _)).trans (le_max_left _ _)
    · exact (le_max_right _ _).trans (le_max_left _ _)
    · exact le_max_right _ _
  · exact max_le (max_le (max_le (le_maxOver f 0) (le_maxOver f 1)) (le_maxOver f 2))
      (le_maxOver f 3)

theorem minOver_fin_two (f : Fin 2 → K) : minOver f = min (f 0) (f 1) := by
  apply le_antisymm
  · exact le_min (minOver_le f 0) (minOver_le f 1)
  · apply le_minOver
    intro i
    fin_cases i
    · exact min_le_left _ _
    · exact min_le_right _ _

theorem minOver_fin_three (f : Fin 3 → K) : minOver f = min (min (f 0) (f 1)) (f 2) := by
  apply le_antisymm
  · exact le_min (le_min (minOver_le f 0) (minOver_le f 1)) (minOver_le f 2)
  · apply le_minOver
    intro i
    fin_cases i
    · exact (min_le_left _ _).trans (min_le_left _ _)
    · exact (min_le_left _ _).trans (min_le_right _ _)
    · exact min_le_right _ _

theorem minOver_fin_four (f : Fin 4 → K) :
    minOver f = min (min (min (f 0) (f 1)) (f 2)) (f 3) := by
  apply le_antisymm
  · exact le_min (le_min (le_min (minOver_le f 0) (minOver_le f 1)) (minOver_le f 2))
      (minOver_le f 3)
  · apply le_minOver
    intro i
    fin_cases i
    · exact ((min_le_left _ _).trans (min_le_left _ _)).trans (min_le_left _ _)
    · exact ((min_le_left _ _).trans (min_le_left _ _)).trans (min_le_right _ _)
    · exact (min_le_left _ _).trans (min_le_right _ _)
    · exact min_le_right _ _

theorem forall_fin_three {P : Fin 3 → Prop} : (∀ i, P i) ↔ P 0 ∧ P 1 ∧ P 2 := by
  constructor
  · intro h
    exact ⟨h 0, h 1, h 2⟩
  · rintro ⟨h0, h1, h2⟩ i
    fin_cases i
    · exact h0
    · exact h1
    · exact h2

end Small

section Field

variable {ι κ K : Type} [Fintype ι] [Nonempty ι] [Fintype κ]
  [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- A nonnegative rescaling and a shift commute with the maximum. -/
theorem maxOver_mul_add (f : ι → K) {s : K} (hs : 0 ≤ s) (c : K) :
    maxOver (fun i => s * f i + c) = s * maxOver f + c := by
  obtain ⟨i, hi⟩ := exists_eq_maxOver f
  refine maxOver_eq_of i (by rw [hi]) fun j => ?_
  have h := mul_le_mul_of_nonneg_left (le_maxOver f j) hs
  linarith

theorem maxOver_mul_left (f : ι → K) {s : K} (hs : 0 ≤ s) :
    maxOver (fun i => s * f i) = s * maxOver f := by
  have h := maxOver_mul_add f hs 0
  simpa using h

/-- `min_i (c - f i) = c - max_i f i`. -/
theorem minOver_const_sub (f : ι → K) (c : K) :
    minOver (fun i => c - f i) = c - maxOver f := by
  obtain ⟨i, hi⟩ := exists_eq_maxOver f
  refine minOver_eq_of i (by rw [hi]) fun j => ?_
  have h := le_maxOver f j
  linarith

/-- The maximum of a sum is at most the sum of the maxima: the inequality behind every
nonnegativity result for the value of information. -/
theorem maxOver_sum_le (W : κ → ι → K) :
    maxOver (fun i => ∑ k, W k i) ≤ ∑ k, maxOver (W k) :=
  maxOver_le fun i => Finset.sum_le_sum fun k _ => le_maxOver (W k) i

/-- Weighted form of `maxOver_sum_le`: a nonnegative mixture of objectives is at most the
mixture of their maxima ("a stochastic decision rule is no better than the best deterministic
rule"). -/
theorem maxOver_weighted_sum_le (w : κ → K) (hw : ∀ k, 0 ≤ w k) (W : κ → ι → K) :
    maxOver (fun i => ∑ k, w k * W k i) ≤ ∑ k, w k * maxOver (W k) :=
  maxOver_le fun i => Finset.sum_le_sum fun k _ =>
    mul_le_mul_of_nonneg_left (le_maxOver (W k) i) (hw k)

end Field

end DecisionAnalysisProofs
