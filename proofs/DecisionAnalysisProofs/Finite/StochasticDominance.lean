import DecisionAnalysisProofs.Finite.Problem
import Mathlib.Analysis.Convex.Function
import Mathlib.Tactic.FieldSimp

/-!
# DecisionAnalysisProofs.Finite.StochasticDominance

First- and second-order stochastic dominance (`fsd_dominates`, `ssd_dominates`,
`StochasticDominance` in `src/deep_uncertainty.jl`, tested in `test/test_track3.jl`).

## Conventions

* A discrete lottery `(x, p)` consists of outcomes `x : Θ → K` on a finite state type `Θ` and
  probability weights `p : Θ → K` (`NonnegPrior p`, `NormalisedPrior p`), over any linearly
  ordered field `K` (instantiated at `ℝ` for theorems and `ℚ` for exact Julia checks).
* `cdf x p t = ∑ θ, if x θ ≤ t then p θ else 0` is the cumulative distribution function `F_x(t)`
  computed by `_joint_cdfs`.
* `integratedCdf x p z = ∑ θ, p θ * max 0 (z - x θ)` is the exact integrated CDF
  `∫_{-∞}^z F_x(t) dt` computed piecewise-linearly by `_ssd_check`.
* `FSD x p y q` is `∀ t, cdf x p t ≤ cdf y q t`; `SSD x p y q` is
  `∀ z, integratedCdf x p z ≤ integratedCdf y q z`. Strict variants add an existential witness
  of strict inequality.
* `IsFSDUndominated` and `IsSSDUndominated` are the sets retained by
  `decide(problem, StochasticDominance(; order = 1))` and `order = 2`.

## Headline results

* **`fsd_refl`**, **`fsd_trans`**, **`strictFSD_irrefl`**, **`strictFSD_trans`**,
  **`ssd_refl`**, **`ssd_trans`**, **`strictSSD_irrefl`**, **`strictSSD_trans`**: both orders
  are preorders and their strict versions are strict partial orders.
* **`fsd_of_pointwise_le`**: statewise dominance implies FSD.
* **`expected_le_of_fsd`**: FSD implies larger expected utility for every monotone utility
  function `g : K → K`.
* **`ssd_of_fsd`**, **`strictSSD_of_strictFSD`**: FSD implies SSD, and strict FSD implies
  strict SSD (for normalised nonnegative lotteries).
* **`mean_le_of_fsd`**, **`mean_le_of_ssd`**: both FSD and SSD imply a weakly larger mean.
* **`cvarObj_le_of_ssd`**: SSD implies a weakly larger Rockafellar–Uryasev CVaR objective at
  every tail level `α > 0` and every threshold `t`, backing the docstring equivalence with
  `CVaR(α)`.
* **`expected_le_of_ssd`**: SSD implies larger expected utility for every non-decreasing concave
  utility function `g : K → K`.
* **`IsSSDUndominated.isFSDUndominated`**: the SSD-efficient set is a subset of the
  FSD-efficient set (`order = 2 ⊆ order = 1`, backing `test/test_track3.jl`).
* **`exists_isFSDUndominated`**, **`exists_isSSDUndominated`**: on any finite nonempty action
  set, `StochasticDominance` always returns at least one action.
* **`isSSDUndominated_of_unique_bayes`**: a uniquely Bayes-optimal action is never strictly
  SSD- or FSD-dominated.

Not claimed: floating-point tolerance (`atol`) behaviour in `_fsd_check` / `_ssd_check`.
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Θ' Θ'' : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Θ'] [Fintype Θ'']
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### Definitions -/

/-- Cumulative distribution function `F_x(t) = ∑_{x θ ≤ t} p θ`. -/
def cdf (x : Θ → K) (p : Θ → K) (t : K) : K :=
  ∑ θ, if x θ ≤ t then p θ else 0

/-- Integrated CDF `∫_{-∞}^z F_x(t) dt = ∑_θ p θ · max(0, z - x θ)`. -/
def integratedCdf (x : Θ → K) (p : Θ → K) (z : K) : K :=
  ∑ θ, p θ * max 0 (z - x θ)

/-- Weak first-order stochastic dominance: `(x, p)` FSD-dominates `(y, q)`. -/
def FSD (x : Θ → K) (p : Θ → K) (y : Θ' → K) (q : Θ' → K) : Prop :=
  ∀ t, cdf x p t ≤ cdf y q t

/-- Strict first-order stochastic dominance (`fsd_dominates(..., strict=true)`). -/
def StrictFSD (x : Θ → K) (p : Θ → K) (y : Θ' → K) (q : Θ' → K) : Prop :=
  FSD x p y q ∧ ∃ t, cdf x p t < cdf y q t

/-- Weak second-order stochastic dominance: `(x, p)` SSD-dominates `(y, q)`. -/
def SSD (x : Θ → K) (p : Θ → K) (y : Θ' → K) (q : Θ' → K) : Prop :=
  ∀ z, integratedCdf x p z ≤ integratedCdf y q z

/-- Strict second-order stochastic dominance (`ssd_dominates(..., strict=true)`). -/
def StrictSSD (x : Θ → K) (p : Θ → K) (y : Θ' → K) (q : Θ' → K) : Prop :=
  SSD x p y q ∧ ∃ z, integratedCdf x p z < integratedCdf y q z

/-- Rockafellar–Uryasev threshold objective for `CVaR(α)` at threshold `t`. -/
def cvarObj (x : Θ → K) (p : Θ → K) (α t : K) : K :=
  t - integratedCdf x p t / α

/-- `a` is retained by `StochasticDominance(; order = 1)`. -/
def IsFSDUndominated (u : A → Θ → K) (p : Θ → K) (a : A) : Prop :=
  ¬ ∃ b, StrictFSD (u b) p (u a) p

/-- `a` is retained by `StochasticDominance(; order = 2)`. -/
def IsSSDUndominated (u : A → Θ → K) (p : Θ → K) (a : A) : Prop :=
  ¬ ∃ b, StrictSSD (u b) p (u a) p

/-! ### Order properties -/

theorem fsd_refl (x : Θ → K) (p : Θ → K) : FSD x p x p := fun _ => le_rfl

theorem fsd_trans {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K} {z : Θ'' → K} {r : Θ'' → K}
    (h1 : FSD x p y q) (h2 : FSD y q z r) : FSD x p z r := fun t => (h1 t).trans (h2 t)

theorem strictFSD_irrefl (x : Θ → K) (p : Θ → K) : ¬ StrictFSD x p x p := by
  rintro ⟨_, t, ht⟩
  exact lt_irrefl _ ht

theorem strictFSD_trans {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    {z : Θ'' → K} {r : Θ'' → K} (h1 : StrictFSD x p y q) (h2 : StrictFSD y q z r) :
    StrictFSD x p z r := by
  refine ⟨fsd_trans h1.1 h2.1, ?_⟩
  obtain ⟨t, ht⟩ := h1.2
  exact ⟨t, lt_of_lt_of_le ht (h2.1 t)⟩

theorem ssd_refl (x : Θ → K) (p : Θ → K) : SSD x p x p := fun _ => le_rfl

theorem ssd_trans {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K} {z : Θ'' → K} {r : Θ'' → K}
    (h1 : SSD x p y q) (h2 : SSD y q z r) : SSD x p z r := fun t => (h1 t).trans (h2 t)

theorem strictSSD_irrefl (x : Θ → K) (p : Θ → K) : ¬ StrictSSD x p x p := by
  rintro ⟨_, z, hz⟩
  exact lt_irrefl _ hz

theorem strictSSD_trans {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    {z : Θ'' → K} {r : Θ'' → K} (h1 : StrictSSD x p y q) (h2 : StrictSSD y q z r) :
    StrictSSD x p z r := by
  refine ⟨ssd_trans h1.1 h2.1, ?_⟩
  obtain ⟨z0, hz0⟩ := h1.2
  exact ⟨z0, lt_of_lt_of_le hz0 (h2.1 z0)⟩

/-- Statewise dominance implies first-order stochastic dominance for any nonnegative prior. -/
theorem fsd_of_pointwise_le {x y : Θ → K} {p : Θ → K} (hp : NonnegPrior p)
    (hxy : ∀ θ, y θ ≤ x θ) : FSD x p y p := by
  intro t
  refine Finset.sum_le_sum fun θ _ => ?_
  by_cases hx : x θ ≤ t
  · have hy : y θ ≤ t := (hxy θ).trans hx
    simp [hx, hy]
  · by_cases hy : y θ ≤ t
    · simp [hx, hy, hp θ]
    · simp [hx, hy]

/-! ### Expected-utility soundness of FSD -/

private theorem cdf_min_eq_of_lt (x : Θ → K) (p : Θ → K) (m' t : K) (ht : t < m') :
    cdf (fun θ => min (x θ) m') p t = cdf x p t := by
  unfold cdf
  refine Finset.sum_congr rfl fun θ _ => ?_
  have hiff : min (x θ) m' ≤ t ↔ x θ ≤ t := by
    simp [not_le.mpr ht]
  simp [hiff]

private theorem cdf_min_eq_sum_of_le (x : Θ → K) (p : Θ → K) (m' t : K) (ht : m' ≤ t) :
    cdf (fun θ => min (x θ) m') p t = ∑ θ, p θ := by
  unfold cdf
  refine Finset.sum_congr rfl fun θ _ => ?_
  have hle : min (x θ) m' ≤ t := (min_le_right _ _).trans ht
  simp [hle]

private theorem fsd_min_cap {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : FSD x p y q) (m' : K) :
    FSD (fun θ => min (x θ) m') p (fun θ' => min (y θ') m') q := by
  intro t
  by_cases ht : t < m'
  · rw [cdf_min_eq_of_lt x p m' t ht, cdf_min_eq_of_lt y q m' t ht]
    exact h t
  · have hle : m' ≤ t := not_lt.mp ht
    rw [cdf_min_eq_sum_of_le x p m' t hle, cdf_min_eq_sum_of_le y q m' t hle, hp1, hq1]

private theorem sum_diff_cap {x : Θ → K} {p : Θ → K} (hp1 : NormalisedPrior p)
    {g : K → K} {S : Finset K} {m m' : K} (hxS : ∀ θ, x θ ∈ S)
    (hm' : ∀ a ∈ S, a ≤ m' ∨ a = m) (hmlt : m' < m) :
    ∑ θ, p θ * g (x θ) - ∑ θ, p θ * g (min (x θ) m') =
      (g m - g m') * (1 - cdf x p m') := by
  have h1 : 1 - cdf x p m' = ∑ θ, if x θ ≤ m' then 0 else p θ := by
    have hsplit : ∑ θ, p θ = (∑ θ, if x θ ≤ m' then p θ else 0) +
        ∑ θ, if x θ ≤ m' then 0 else p θ := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun θ _ => ?_
      split_ifs <;> ring
    unfold cdf
    unfold NormalisedPrior at hp1
    linarith
  rw [h1, Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun θ _ => ?_
  rcases hm' (x θ) (hxS θ) with hle | heq
  · rw [if_pos hle, min_eq_left hle, mul_zero, sub_self]
  · rw [heq, if_neg (not_le.mpr hmlt), min_eq_right hmlt.le]
    ring

private theorem expected_le_of_fsd_aux :
    ∀ n : ℕ, ∀ (S : Finset K), S.card ≤ n →
      ∀ {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K},
        NormalisedPrior p → NormalisedPrior q →
        (∀ θ, x θ ∈ S) → (∀ θ', y θ' ∈ S) →
        FSD x p y q → ∀ {g : K → K}, Monotone g →
        ∑ θ', q θ' * g (y θ') ≤ ∑ θ, p θ * g (x θ) := by
  intro n
  induction n with
  | zero =>
    intro S hS x p y q hp1 hq1 hxS hyS _ g _
    have hSe : S = ∅ := Finset.card_eq_zero.mp (Nat.eq_zero_of_le_zero hS)
    have hxe : IsEmpty Θ := ⟨fun θ => by have := hxS θ; simp [hSe] at this⟩
    have hye : IsEmpty Θ' := ⟨fun θ' => by have := hyS θ'; simp [hSe] at this⟩
    simp
  | succ n ih =>
    intro S hS x p y q hp1 hq1 hxS hyS hfsd g hg
    by_cases hSne : S.Nonempty
    · let m := S.max' hSne
      have hmS : m ∈ S := Finset.max'_mem S hSne
      let S' := S.erase m
      by_cases hS'ne : S'.Nonempty
      · let m' := S'.max' hS'ne
        have hm'S' : m' ∈ S' := Finset.max'_mem S' hS'ne
        have hm'lt : m' < m := by
          have hle : m' ≤ m := Finset.le_max' S m' (Finset.mem_of_mem_erase hm'S')
          have hne : m' ≠ m := (Finset.mem_erase.mp hm'S').1
          exact lt_of_le_of_ne hle hne
        have hpart : ∀ a ∈ S, a ≤ m' ∨ a = m := by
          intro a ha
          by_cases ham : a = m
          · exact Or.inr ham
          · exact Or.inl (Finset.le_max' S' a (Finset.mem_erase.mpr ⟨ham, ha⟩))
        have hS'card : S'.card ≤ n := by
          have hc : S'.card = S.card - 1 := Finset.card_erase_of_mem hmS
          have hpos : 1 ≤ S.card := Finset.card_pos.mpr hSne
          omega
        have hx'S' : ∀ θ, min (x θ) m' ∈ S' := by
          intro θ
          rcases hpart (x θ) (hxS θ) with hle | heq
          · rw [min_eq_left hle]
            exact Finset.mem_erase.mpr ⟨(lt_of_le_of_lt hle hm'lt).ne, hxS θ⟩
          · rw [heq, min_eq_right hm'lt.le]
            exact hm'S'
        have hy'S' : ∀ θ', min (y θ') m' ∈ S' := by
          intro θ'
          rcases hpart (y θ') (hyS θ') with hle | heq
          · rw [min_eq_left hle]
            exact Finset.mem_erase.mpr ⟨(lt_of_le_of_lt hle hm'lt).ne, hyS θ'⟩
          · rw [heq, min_eq_right hm'lt.le]
            exact hm'S'
        have hbase := ih S' hS'card hp1 hq1 hx'S' hy'S' (fsd_min_cap hp1 hq1 hfsd m') hg
        have hdx := sum_diff_cap hp1 hxS hpart hm'lt (g := g)
        have hdy := sum_diff_cap hq1 hyS hpart hm'lt (g := g)
        have hgm : 0 ≤ g m - g m' := sub_nonneg.mpr (hg hm'lt.le)
        have hcdf : 1 - cdf y q m' ≤ 1 - cdf x p m' := by linarith [hfsd m']
        have hstep : (g m - g m') * (1 - cdf y q m') ≤ (g m - g m') * (1 - cdf x p m') :=
          mul_le_mul_of_nonneg_left hcdf hgm
        linarith
      · -- Every element of `S` equals `m`.
        have hall : ∀ a ∈ S, a = m := by
          intro a ha
          by_contra ham
          exact hS'ne ⟨a, Finset.mem_erase.mpr ⟨ham, ha⟩⟩
        have hxe : ∑ θ, p θ * g (x θ) = g m := by
          have : ∀ θ, p θ * g (x θ) = p θ * g m := fun θ => by rw [hall (x θ) (hxS θ)]
          rw [Finset.sum_congr rfl fun θ _ => this θ, ← Finset.sum_mul, hp1, one_mul]
        have hye : ∑ θ', q θ' * g (y θ') = g m := by
          have : ∀ θ', q θ' * g (y θ') = q θ' * g m := fun θ' => by rw [hall (y θ') (hyS θ')]
          rw [Finset.sum_congr rfl fun θ' _ => this θ', ← Finset.sum_mul, hq1, one_mul]
        rw [hxe, hye]
    · have hSe : S = ∅ := Finset.not_nonempty_iff_eq_empty.mp hSne
      have hxe : IsEmpty Θ := ⟨fun θ => by have := hxS θ; simp [hSe] at this⟩
      have hye : IsEmpty Θ' := ⟨fun θ' => by have := hyS θ'; simp [hSe] at this⟩
      simp

/-- **FSD expected-utility soundness.** If `(x, p)` FSD-dominates `(y, q)` (both normalised),
then every non-decreasing utility function `g` gives `(x, p)` at least the expected utility
of `(y, q)`. -/
theorem expected_le_of_fsd {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : FSD x p y q)
    {g : K → K} (hg : Monotone g) :
    ∑ θ', q θ' * g (y θ') ≤ ∑ θ, p θ * g (x θ) := by
  classical
  let S : Finset K := Finset.univ.image x ∪ Finset.univ.image y
  exact expected_le_of_fsd_aux S.card S le_rfl hp1 hq1
    (fun θ => Finset.mem_union_left _ (Finset.mem_image_of_mem x (Finset.mem_univ θ)))
    (fun θ' => Finset.mem_union_right _ (Finset.mem_image_of_mem y (Finset.mem_univ θ')))
    h hg

/-- FSD implies a weakly larger expected value. -/
theorem mean_le_of_fsd {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : FSD x p y q) :
    ∑ θ', q θ' * y θ' ≤ ∑ θ, p θ * x θ :=
  expected_le_of_fsd hp1 hq1 h monotone_id

/-- First-order stochastic dominance implies second-order stochastic dominance. -/
theorem ssd_of_fsd {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : FSD x p y q) :
    SSD x p y q := by
  intro z
  have hg : Monotone (fun t : K => -max 0 (z - t)) := by
    intro a b hab
    have : max 0 (z - b) ≤ max 0 (z - a) := max_le_max le_rfl (sub_le_sub_left hab z)
    linarith
  have hle := expected_le_of_fsd hp1 hq1 h hg
  simp only [mul_neg, Finset.sum_neg_distrib] at hle
  unfold integratedCdf
  linarith

/-- Strict FSD implies strict SSD for normalised nonnegative lotteries. -/
theorem strictSSD_of_strictFSD {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : StrictFSD x p y q) :
    StrictSSD x p y q := by
  classical
  refine ⟨ssd_of_fsd hp1 hq1 h.1, ?_⟩
  obtain ⟨t0, ht0⟩ := h.2
  let S : Finset K := (Finset.univ.image x ∪ Finset.univ.image y).filter (t0 < ·)
  have hSne : S.Nonempty := by
    by_contra hne
    have hle_x : ∀ θ, x θ ≤ t0 := by
      intro θ
      by_contra hlt
      exact hne ⟨x θ, Finset.mem_filter.mpr
        ⟨Finset.mem_union_left _ (Finset.mem_image_of_mem x (Finset.mem_univ θ)),
          not_le.mp hlt⟩⟩
    have hle_y : ∀ θ', y θ' ≤ t0 := by
      intro θ'
      by_contra hlt
      exact hne ⟨y θ', Finset.mem_filter.mpr
        ⟨Finset.mem_union_right _ (Finset.mem_image_of_mem y (Finset.mem_univ θ')),
          not_le.mp hlt⟩⟩
    have hcx : cdf x p t0 = 1 := by
      unfold cdf
      rw [Finset.sum_congr rfl fun θ _ => if_pos (hle_x θ)]
      exact hp1
    have hcy : cdf y q t0 = 1 := by
      unfold cdf
      rw [Finset.sum_congr rfl fun θ' _ => if_pos (hle_y θ')]
      exact hq1
    linarith
  let z := S.min' hSne
  have hz_gt : t0 < z := (Finset.mem_filter.mp (Finset.min'_mem S hSne)).2
  have hsplit : ∀ a : K, (a ≤ t0 ∨ z ≤ a) →
      max 0 (z - a) = (z - t0) * (if a ≤ t0 then 1 else 0) + max 0 (t0 - a) := by
    intro a ha
    rcases ha with hle | hge
    · rw [if_pos hle, mul_one, max_eq_right (by linarith : (0 : K) ≤ z - a),
        max_eq_right (by linarith : (0 : K) ≤ t0 - a)]
      ring
    · have hgt : ¬ a ≤ t0 := not_le.mpr (lt_of_lt_of_le hz_gt hge)
      rw [if_neg hgt, mul_zero, zero_add,
        max_eq_left (by linarith : z - a ≤ 0),
        max_eq_left (by linarith : t0 - a ≤ 0)]
  have hx_part : ∀ θ, x θ ≤ t0 ∨ z ≤ x θ := by
    intro θ
    by_cases hle : x θ ≤ t0
    · exact Or.inl hle
    · exact Or.inr (Finset.min'_le S (x θ) (Finset.mem_filter.mpr
        ⟨Finset.mem_union_left _ (Finset.mem_image_of_mem x (Finset.mem_univ θ)),
          not_le.mp hle⟩))
  have hy_part : ∀ θ', y θ' ≤ t0 ∨ z ≤ y θ' := by
    intro θ'
    by_cases hle : y θ' ≤ t0
    · exact Or.inl hle
    · exact Or.inr (Finset.min'_le S (y θ') (Finset.mem_filter.mpr
        ⟨Finset.mem_union_right _ (Finset.mem_image_of_mem y (Finset.mem_univ θ')),
          not_le.mp hle⟩))
  have hix : integratedCdf x p z = (z - t0) * cdf x p t0 + integratedCdf x p t0 := by
    unfold integratedCdf cdf
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun θ _ => ?_
    rw [hsplit (x θ) (hx_part θ)]
    split_ifs <;> ring
  have hiy : integratedCdf y q z = (z - t0) * cdf y q t0 + integratedCdf y q t0 := by
    unfold integratedCdf cdf
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun θ' _ => ?_
    rw [hsplit (y θ') (hy_part θ')]
    split_ifs <;> ring
  refine ⟨z, ?_⟩
  rw [hix, hiy]
  have h1 : (z - t0) * cdf x p t0 < (z - t0) * cdf y q t0 :=
    mul_lt_mul_of_pos_left ht0 (sub_pos.mpr hz_gt)
  have h2 : integratedCdf x p t0 ≤ integratedCdf y q t0 := ssd_of_fsd hp1 hq1 h.1 t0
  linarith

/-! ### Mean, CVaR, and concave expected-utility soundness of SSD -/

/-- SSD implies a weakly larger expected value. -/
theorem mean_le_of_ssd {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : SSD x p y q) :
    ∑ θ', q θ' * y θ' ≤ ∑ θ, p θ * x θ := by
  classical
  let S : Finset K := (Finset.univ.image x ∪ Finset.univ.image y) ∪ {0}
  have hSne : S.Nonempty := ⟨0, Finset.mem_union_right _ (Finset.mem_singleton_self 0)⟩
  let z := S.max' hSne
  have hzx : ∀ θ, x θ ≤ z := fun θ =>
    Finset.le_max' S (x θ)
      (Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_image_of_mem x (Finset.mem_univ θ))))
  have hzy : ∀ θ', y θ' ≤ z := fun θ' =>
    Finset.le_max' S (y θ')
      (Finset.mem_union_left _ (Finset.mem_union_right _ (Finset.mem_image_of_mem y (Finset.mem_univ θ'))))
  have hix : integratedCdf x p z = z - ∑ θ, p θ * x θ := by
    unfold integratedCdf
    have : ∀ θ, p θ * max 0 (z - x θ) = p θ * z - p θ * x θ := fun θ => by
      rw [max_eq_right (sub_nonneg.mpr (hzx θ))]; ring
    rw [Finset.sum_congr rfl fun θ _ => this θ, Finset.sum_sub_distrib, ← Finset.sum_mul,
      hp1, one_mul]
  have hiy : integratedCdf y q z = z - ∑ θ', q θ' * y θ' := by
    unfold integratedCdf
    have : ∀ θ', q θ' * max 0 (z - y θ') = q θ' * z - q θ' * y θ' := fun θ' => by
      rw [max_eq_right (sub_nonneg.mpr (hzy θ'))]; ring
    rw [Finset.sum_congr rfl fun θ' _ => this θ', Finset.sum_sub_distrib, ← Finset.sum_mul,
      hq1, one_mul]
  have hz := h z
  rw [hix, hiy] at hz
  linarith

/-- **SSD implies higher CVaR at every tail level.** At every `α > 0` and every Rockafellar–
Uryasev threshold `t`, the objective for `(x, p)` is at least that for `(y, q)`. -/
theorem cvarObj_le_of_ssd {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (h : SSD x p y q) {α : K} (hα : 0 < α) (t : K) :
    cvarObj y q α t ≤ cvarObj x p α t := by
  unfold cvarObj
  have hdiv : integratedCdf x p t / α ≤ integratedCdf y q t / α :=
    div_le_div_of_nonneg_right (h t) hα.le
  linarith

private theorem ssd_min_cap {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : SSD x p y q) (m' : K) :
    SSD (fun θ => min (x θ) m') p (fun θ' => min (y θ') m') q := by
  intro z
  by_cases hz : z ≤ m'
  · have hEq : ∀ a : K, max 0 (z - min a m') = max 0 (z - a) := by
      intro a
      by_cases ha : a ≤ m'
      · rw [min_eq_left ha]
      · rw [min_eq_right (le_of_not_ge ha), max_eq_left (by linarith : z - m' ≤ 0),
          max_eq_left (by linarith : z - a ≤ 0)]
    unfold integratedCdf
    simp only [hEq]
    exact h z
  · have hgt : m' < z := not_le.mp hz
    have hEq : ∀ a : K, max 0 (z - min a m') = (z - m') + max 0 (m' - a) := by
      intro a
      by_cases ha : a ≤ m'
      · rw [min_eq_left ha, max_eq_right (by linarith : (0 : K) ≤ z - a),
          max_eq_right (by linarith : (0 : K) ≤ m' - a)]
        ring
      · rw [min_eq_right (le_of_not_ge ha), max_eq_right (by linarith : (0 : K) ≤ z - m'),
          max_eq_left (by linarith : m' - a ≤ 0), add_zero]
    have hix : integratedCdf (fun θ => min (x θ) m') p z = (z - m') + integratedCdf x p m' := by
      unfold integratedCdf
      have : ∀ θ, p θ * max 0 (z - min (x θ) m') = p θ * (z - m') + p θ * max 0 (m' - x θ) :=
        fun θ => by rw [hEq (x θ)]; ring
      rw [Finset.sum_congr rfl fun θ _ => this θ, Finset.sum_add_distrib, ← Finset.sum_mul,
        hp1, one_mul]
    have hiy : integratedCdf (fun θ' => min (y θ') m') q z = (z - m') + integratedCdf y q m' := by
      unfold integratedCdf
      have : ∀ θ', q θ' * max 0 (z - min (y θ') m') =
          q θ' * (z - m') + q θ' * max 0 (m' - y θ') := fun θ' => by rw [hEq (y θ')]; ring
      rw [Finset.sum_congr rfl fun θ' _ => this θ', Finset.sum_add_distrib, ← Finset.sum_mul,
        hq1, one_mul]
    rw [hix, hiy]
    linarith [h m']

private theorem concave_secant_le {g : K → K} (hconc : ConcaveOn K Set.univ g)
    {a b c : K} (hab : a < b) (hbc : b < c) :
    (g c - g b) / (c - b) ≤ (g b - g a) / (b - a) := by
  have hca : 0 < c - a := by linarith
  have hba : 0 < b - a := by linarith
  have hcb : 0 < c - b := by linarith
  have h1 : 0 ≤ (c - b) / (c - a) := div_nonneg hcb.le hca.le
  have h2 : 0 ≤ (b - a) / (c - a) := div_nonneg hba.le hca.le
  have hsum : (c - b) / (c - a) + (b - a) / (c - a) = 1 := by field_simp; ring
  have hcomb : (c - b) / (c - a) * a + (b - a) / (c - a) * c = b := by field_simp; ring
  have hineq := hconc.2 (Set.mem_univ a) (Set.mem_univ c) h1 h2 hsum
  simp only [smul_eq_mul, hcomb] at hineq
  rw [div_le_div_iff₀ hcb hba]
  have hmul : (c - b) * g a + (b - a) * g c ≤ (c - a) * g b := by
    have := mul_le_mul_of_nonneg_left hineq hca.le
    have hlhs : (c - a) * ((c - b) / (c - a) * g a + (b - a) / (c - a) * g c) =
        (c - b) * g a + (b - a) * g c := by field_simp
    linarith
  nlinarith

private theorem expected_le_of_ssd_aux :
    ∀ n : ℕ, ∀ (S : Finset K), S.card ≤ n →
      ∀ {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K},
        NormalisedPrior p → NormalisedPrior q →
        (∀ θ, x θ ∈ S) → (∀ θ', y θ' ∈ S) →
        SSD x p y q → ∀ {g : K → K}, Monotone g → ConcaveOn K Set.univ g →
        ∑ θ', q θ' * g (y θ') ≤ ∑ θ, p θ * g (x θ) := by
  intro n
  induction n with
  | zero =>
    intro S hS x p y q hp1 hq1 hxS hyS _ g _ _
    have hSe : S = ∅ := Finset.card_eq_zero.mp (Nat.eq_zero_of_le_zero hS)
    have hxe : IsEmpty Θ := ⟨fun θ => by have := hxS θ; simp [hSe] at this⟩
    have hye : IsEmpty Θ' := ⟨fun θ' => by have := hyS θ'; simp [hSe] at this⟩
    simp
  | succ n ih =>
    intro S hS x p y q hp1 hq1 hxS hyS hssd g hg hconc
    by_cases hSne : S.Nonempty
    · let m := S.max' hSne
      have hmS : m ∈ S := Finset.max'_mem S hSne
      let S' := S.erase m
      by_cases hS'ne : S'.Nonempty
      · let m' := S'.max' hS'ne
        have hm'S' : m' ∈ S' := Finset.max'_mem S' hS'ne
        have hm'lt : m' < m :=
          lt_of_le_of_ne (Finset.le_max' S m' (Finset.mem_of_mem_erase hm'S'))
            (Finset.mem_erase.mp hm'S').1
        have hpart : ∀ a ∈ S, a ≤ m' ∨ a = m := by
          intro a ha
          by_cases ham : a = m
          · exact Or.inr ham
          · exact Or.inl (Finset.le_max' S' a (Finset.mem_erase.mpr ⟨ham, ha⟩))
        have hS'card : S'.card ≤ n := by
          have hc : S'.card = S.card - 1 := Finset.card_erase_of_mem hmS
          have hpos : 1 ≤ S.card := Finset.card_pos.mpr hSne
          omega
        let s : K := (g m - g m') / (m - m')
        have hs0 : 0 ≤ s := div_nonneg (sub_nonneg.mpr (hg hm'lt.le)) (sub_pos.mpr hm'lt).le
        let gcap : K → K := fun t => g (min t m') - s * min t m'
        have hgcap_mono : Monotone gcap := by
          intro u v huv
          have hab : min u m' ≤ min v m' := min_le_min huv le_rfl
          rcases eq_or_lt_of_le hab with heq | hlt
          · simp [gcap, heq]
          · have hbm' : min v m' ≤ m' := min_le_right v m'
            have hsec : s ≤ (g (min v m') - g (min u m')) / (min v m' - min u m') := by
              rcases eq_or_lt_of_le hbm' with hvm' | hvm'lt
              · rw [hvm']
                rw [hvm'] at hlt
                exact concave_secant_le hconc hlt hm'lt
              · have h1 := concave_secant_le hconc hlt hvm'lt
                have h2 := concave_secant_le hconc hvm'lt hm'lt
                exact h2.trans h1
            have hpos : 0 < min v m' - min u m' := sub_pos.mpr hlt
            have hmul := (le_div_iff₀ hpos).mp hsec
            change g (min u m') - s * min u m' ≤ g (min v m') - s * min v m'
            linarith
        have hgcap_conc : ConcaveOn K Set.univ gcap := by
          refine ⟨convex_univ, fun u _ v _ a b ha hb hab => ?_⟩
          have hmin : a * min u m' + b * min v m' ≤ min (a * u + b * v) m' :=
            le_min (add_le_add (mul_le_mul_of_nonneg_left (min_le_left u m') ha)
              (mul_le_mul_of_nonneg_left (min_le_left v m') hb))
              (by
                calc a * min u m' + b * min v m'
                    ≤ a * m' + b * m' :=
                      add_le_add (mul_le_mul_of_nonneg_left (min_le_right u m') ha)
                        (mul_le_mul_of_nonneg_left (min_le_right v m') hb)
                  _ = m' := by rw [← add_mul, hab, one_mul])
          have hstep1 : gcap (a * min u m' + b * min v m') ≤ gcap (a * u + b * v) := by
            have hle_m' : a * min u m' + b * min v m' ≤ m' := hmin.trans (min_le_right _ _)
            calc gcap (a * min u m' + b * min v m')
                = gcap (min (a * min u m' + b * min v m') m') := by rw [min_eq_left hle_m']
              _ ≤ gcap (min (a * u + b * v) m') := hgcap_mono (by rw [min_eq_left hle_m']; exact hmin)
              _ = gcap (a * u + b * v) := by simp [gcap]
          have hle_m' : a * min u m' + b * min v m' ≤ m' := hmin.trans (min_le_right _ _)
          have hstep2 : a * gcap u + b * gcap v ≤ gcap (a * min u m' + b * min v m') := by
            have hc := hconc.2 (Set.mem_univ (min u m')) (Set.mem_univ (min v m')) ha hb hab
            simp only [smul_eq_mul] at hc
            change a * (g (min u m') - s * min u m') + b * (g (min v m') - s * min v m') ≤
              g (min (a * min u m' + b * min v m') m') - s * min (a * min u m' + b * min v m') m'
            rw [min_eq_left hle_m']
            linarith
          simpa [smul_eq_mul] using hstep2.trans hstep1
        have hdecomp : ∀ a ∈ S, g a = gcap (min a m') + s * a := by
          intro a ha
          rcases hpart a ha with hle | heq
          · simp only [gcap, min_eq_left hle]
            ring
          · rw [heq, min_eq_right hm'lt.le]
            simp only [gcap, min_self]
            have hne : m - m' ≠ 0 := (sub_pos.mpr hm'lt).ne'
            change g m = g m' - (g m - g m') / (m - m') * m' + (g m - g m') / (m - m') * m
            field_simp [hne]
            ring
        have hx'S' : ∀ θ, min (x θ) m' ∈ S' := by
          intro θ
          rcases hpart (x θ) (hxS θ) with hle | heq
          · rw [min_eq_left hle]
            exact Finset.mem_erase.mpr ⟨(lt_of_le_of_lt hle hm'lt).ne, hxS θ⟩
          · rw [heq, min_eq_right hm'lt.le]
            exact hm'S'
        have hy'S' : ∀ θ', min (y θ') m' ∈ S' := by
          intro θ'
          rcases hpart (y θ') (hyS θ') with hle | heq
          · rw [min_eq_left hle]
            exact Finset.mem_erase.mpr ⟨(lt_of_le_of_lt hle hm'lt).ne, hyS θ'⟩
          · rw [heq, min_eq_right hm'lt.le]
            exact hm'S'
        have hbase := ih S' hS'card hp1 hq1 hx'S' hy'S' (ssd_min_cap hp1 hq1 hssd m')
          hgcap_mono hgcap_conc
        have hmean := mean_le_of_ssd hp1 hq1 hssd
        have hlin : s * ∑ θ', q θ' * y θ' ≤ s * ∑ θ, p θ * x θ :=
          mul_le_mul_of_nonneg_left hmean hs0
        have hsumx : ∑ θ, p θ * g (x θ) =
            ∑ θ, p θ * gcap (min (x θ) m') + s * ∑ θ, p θ * x θ := by
          rw [Finset.mul_sum, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun θ _ => ?_
          rw [hdecomp (x θ) (hxS θ)]
          ring
        have hsumy : ∑ θ', q θ' * g (y θ') =
            ∑ θ', q θ' * gcap (min (y θ') m') + s * ∑ θ', q θ' * y θ' := by
          rw [Finset.mul_sum, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun θ' _ => ?_
          rw [hdecomp (y θ') (hyS θ')]
          ring
        linarith
      · have hall : ∀ a ∈ S, a = m := by
          intro a ha
          by_contra ham
          exact hS'ne ⟨a, Finset.mem_erase.mpr ⟨ham, ha⟩⟩
        have hxe : ∑ θ, p θ * g (x θ) = g m := by
          have : ∀ θ, p θ * g (x θ) = p θ * g m := fun θ => by rw [hall (x θ) (hxS θ)]
          rw [Finset.sum_congr rfl fun θ _ => this θ, ← Finset.sum_mul, hp1, one_mul]
        have hye : ∑ θ', q θ' * g (y θ') = g m := by
          have : ∀ θ', q θ' * g (y θ') = q θ' * g m := fun θ' => by rw [hall (y θ') (hyS θ')]
          rw [Finset.sum_congr rfl fun θ' _ => this θ', ← Finset.sum_mul, hq1, one_mul]
        rw [hxe, hye]
    · have hSe : S = ∅ := Finset.not_nonempty_iff_eq_empty.mp hSne
      have hxe : IsEmpty Θ := ⟨fun θ => by have := hxS θ; simp [hSe] at this⟩
      have hye : IsEmpty Θ' := ⟨fun θ' => by have := hyS θ'; simp [hSe] at this⟩
      simp

/-- **SSD expected-utility soundness.** If `(x, p)` SSD-dominates `(y, q)` (both normalised),
then every non-decreasing concave (risk-averse) utility function `g` gives `(x, p)` at least
the expected utility of `(y, q)`. -/
theorem expected_le_of_ssd {x : Θ → K} {p : Θ → K} {y : Θ' → K} {q : Θ' → K}
    (hp1 : NormalisedPrior p) (hq1 : NormalisedPrior q) (h : SSD x p y q)
    {g : K → K} (hg : Monotone g) (hconc : ConcaveOn K Set.univ g) :
    ∑ θ', q θ' * g (y θ') ≤ ∑ θ, p θ * g (x θ) := by
  classical
  let S : Finset K := Finset.univ.image x ∪ Finset.univ.image y
  exact expected_le_of_ssd_aux S.card S le_rfl hp1 hq1
    (fun θ => Finset.mem_union_left _ (Finset.mem_image_of_mem x (Finset.mem_univ θ)))
    (fun θ' => Finset.mem_union_right _ (Finset.mem_image_of_mem y (Finset.mem_univ θ')))
    h hg hconc

/-! ### Properties of the `StochasticDominance` decision rule -/

/-- Every SSD-undominated action is FSD-undominated (`order = 2 ⊆ order = 1`). -/
theorem IsSSDUndominated.isFSDUndominated {u : A → Θ → K} {p : Θ → K}
    (hp1 : NormalisedPrior p) {a : A} (h : IsSSDUndominated u p a) :
    IsFSDUndominated u p a := by
  rintro ⟨b, hb⟩
  exact h ⟨b, strictSSD_of_strictFSD hp1 hp1 hb⟩

private theorem exists_maximal_of_strict_order {r : A → A → Prop}
    (hirr : ∀ a, ¬ r a a) (htrans : ∀ a b c, r a b → r b c → r a c) :
    ∃ a : A, ¬ ∃ b, r b a := by
  classical
  suffices h : ∀ (S : Finset A), S.Nonempty → ∃ a ∈ S, ¬ ∃ b ∈ S, r b a by
    obtain ⟨a, _, ha⟩ := h Finset.univ Finset.univ_nonempty
    exact ⟨a, fun ⟨b, hb⟩ => ha ⟨b, Finset.mem_univ b, hb⟩⟩
  intro S
  induction S using Finset.induction_on with
  | empty => intro h; exact (Finset.not_nonempty_empty h).elim
  | @insert x s hx ih =>
    intro _
    by_cases hs : s.Nonempty
    · obtain ⟨a, has, ha⟩ := ih hs
      by_cases hxa : r x a
      · refine ⟨x, Finset.mem_insert_self x s, ?_⟩
        rintro ⟨b, hb, hbx⟩
        rcases Finset.mem_insert.mp hb with rfl | hbs
        · exact hirr b hbx
        · exact ha ⟨b, hbs, htrans b x a hbx hxa⟩
      · refine ⟨a, Finset.mem_insert_of_mem has, ?_⟩
        rintro ⟨b, hb, hba⟩
        rcases Finset.mem_insert.mp hb with rfl | hbs
        · exact hxa hba
        · exact ha ⟨b, hbs, hba⟩
    · have hse : s = ∅ := Finset.not_nonempty_iff_eq_empty.mp hs
      subst hse
      refine ⟨x, Finset.mem_singleton_self x, ?_⟩
      rintro ⟨b, hb, hbx⟩
      rw [Finset.mem_singleton.mp hb] at hbx
      exact hirr x hbx

/-- `StochasticDominance(; order = 1)` always retains at least one action. -/
theorem exists_isFSDUndominated (u : A → Θ → K) (p : Θ → K) :
    ∃ a, IsFSDUndominated u p a :=
  exists_maximal_of_strict_order
    (fun a => strictFSD_irrefl (u a) p)
    (fun _ _ _ h1 h2 => strictFSD_trans h1 h2)

/-- `StochasticDominance(; order = 2)` always retains at least one action. -/
theorem exists_isSSDUndominated (u : A → Θ → K) (p : Θ → K) :
    ∃ a, IsSSDUndominated u p a :=
  exists_maximal_of_strict_order
    (fun a => strictSSD_irrefl (u a) p)
    (fun _ _ _ h1 h2 => strictSSD_trans h1 h2)

/-- A uniquely Bayes-optimal action is always SSD-undominated (and therefore FSD-undominated). -/
theorem isSSDUndominated_of_unique_bayes {u : A → Θ → K} {p : Θ → K}
    (hp1 : NormalisedPrior p) {a : A}
    (huniq : ∀ b, b ≠ a → expectedUtility u p b < expectedUtility u p a) :
    IsSSDUndominated u p a := by
  rintro ⟨b, hb⟩
  by_cases hba : b = a
  · subst hba
    exact strictSSD_irrefl (u b) p hb
  · have hmean : expectedUtility u p a ≤ expectedUtility u p b := mean_le_of_ssd hp1 hp1 hb.1
    linarith [huniq b hba]

end DecisionAnalysisProofs
