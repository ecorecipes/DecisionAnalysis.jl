import DecisionAnalysisProofs.Finite.Fidelity

/-!
# DecisionAnalysisProofs.Finite.InfoGap

Exact total-variation info-gap robustness and opportuneness (`info_gap_robustness`,
`info_gap_opportuneness`, `_tv_ball_robustness`, `_tv_ball_opportuneness` in
`src/deep_uncertainty.jl`, tested in `test/test_track3.jl`).

## Conventions

* A nominal belief `p₀ : Θ → K` (`NonnegPrior p₀`, `NormalisedPrior p₀`) and action payoff row
  `q : Θ → K` live on a finite nonempty state type `Θ` over any linearly ordered field `K`.
* The info-gap uncertainty set at horizon `α` is the total-variation ball
  `InTVBall p₀ α p := NonnegPrior p ∧ NormalisedPrior p ∧ totalVariation p p₀ ≤ α`.
* `GuaranteedAt q p₀ r α` holds when every `p` in the `α`-ball achieves expected payoff at
  least `r` (robustness); `AchievableAt q p₀ w α` holds when some `p` in the `α`-ball achieves
  expected payoff at least `w` (opportuneness).
* A feasible mass shift `ShiftFeasible p₀ α δ` satisfies `0 ≤ δ θ ≤ p₀ θ` and `∑ θ, δ θ ≤ α`.

## Headline results

* **`guaranteedAt_antitone`**, **`achievableAt_monotone`**: robustness is antitone and
  opportuneness is monotone in the horizon `α` (backing monotone bisection and nested horizon
  semantics).
* **`guaranteedAt_zero_iff`**, **`guaranteedAt_one_iff`**: at `α = 0`, robustness holds iff
  the nominal expectation meets `r` (`nom < requirement → 0`); at `α = 1`, iff the worst state
  `minOver q` meets `r` (`qmin >= requirement → 1`).
* **`achievableAt_zero_iff`**, **`achievableAt_one_iff`**: at `α = 0`, opportuneness holds iff
  the nominal expectation meets `w` (`nom >= windfall → 0`); at `α = 1`, iff `maxOver q` meets
  `w` (`qmax < windfall → nothing`).
* **`tvBall_drop_le`** and **`exists_tvBall_attaining_shift`**: the exact worst-case expected
  payoff drop over `InTVBall p₀ α` equals `max { ∑ δ θ * (q θ - minOver q) }` over feasible
  mass shifts `ShiftFeasible p₀ α δ`.
* **`tvBall_gain_le`** and **`exists_tvBall_attaining_gain_shift`**: the exact best-case
  expected payoff gain over `InTVBall p₀ α` equals `max { ∑ δ θ * (maxOver q - q θ) }` over
  feasible mass shifts `ShiftFeasible p₀ α δ`.
* **`greedy_drop_shift_optimal`** and **`greedy_gain_shift_optimal`**: any threshold-saturating
  shift (exhausting highest-payoff states first for robustness, lowest-payoff states first for
  opportuneness) maximises the drop/gain among all shifts of equal or smaller mass — proving
  that Julia's greedy exact inversion in `_tv_ball_robustness` and `_tv_ball_opportuneness` is
  globally optimal over the total-variation ball.

Not claimed: floating-point bisection tolerances in the callable forms of
`info_gap_robustness` / `info_gap_opportuneness`.
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {Θ : Type} [Fintype Θ] [Nonempty Θ]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### Total-variation ball and info-gap predicates -/

/-- The total-variation ball `U(α, p₀) = { p ∈ Δ(Θ) : TV(p, p₀) ≤ α }`. -/
def InTVBall (p₀ : Θ → K) (α : K) (p : Θ → K) : Prop :=
  NonnegPrior p ∧ NormalisedPrior p ∧ totalVariation p p₀ ≤ α

/-- Every belief in the `α`-ball around `p₀` guarantees expected payoff at least `r`. -/
def GuaranteedAt (q p₀ : Θ → K) (r α : K) : Prop :=
  ∀ p, InTVBall p₀ α p → r ≤ ∑ θ, p θ * q θ

/-- Some belief in the `α`-ball around `p₀` achieves expected payoff at least `w`. -/
def AchievableAt (q p₀ : Θ → K) (w α : K) : Prop :=
  ∃ p, InTVBall p₀ α p ∧ w ≤ ∑ θ, p θ * q θ

/-- A feasible mass shift of size at most `α` out of `p₀`: `0 ≤ δ θ ≤ p₀ θ` and `∑ δ ≤ α`. -/
def ShiftFeasible (p₀ : Θ → K) (α : K) (δ : Θ → K) : Prop :=
  (∀ θ, 0 ≤ δ θ) ∧ (∀ θ, δ θ ≤ p₀ θ) ∧ ∑ θ, δ θ ≤ α

/-! ### Monotonicity and boundary characterisations -/

theorem inTVBall_mono {p₀ : Θ → K} {α α' : K} (hα : α ≤ α') {p : Θ → K}
    (hp : InTVBall p₀ α p) : InTVBall p₀ α' p :=
  ⟨hp.1, hp.2.1, hp.2.2.trans hα⟩

/-- Robustness is antitone in the uncertainty horizon `α`. -/
theorem guaranteedAt_antitone {q p₀ : Θ → K} {r α α' : K} (hα : α ≤ α')
    (h : GuaranteedAt q p₀ r α') : GuaranteedAt q p₀ r α :=
  fun p hp => h p (inTVBall_mono hα hp)

/-- Opportuneness is monotone in the uncertainty horizon `α`. -/
theorem achievableAt_monotone {q p₀ : Θ → K} {w α α' : K} (hα : α ≤ α')
    (h : AchievableAt q p₀ w α) : AchievableAt q p₀ w α' := by
  obtain ⟨p, hp, hval⟩ := h
  exact ⟨p, inTVBall_mono hα hp, hval⟩

/-- At horizon `α = 0`, robustness holds iff the nominal expected payoff meets `r`. -/
theorem guaranteedAt_zero_iff {q p₀ : Θ → K} (hp0 : NonnegPrior p₀) (hp1 : NormalisedPrior p₀)
    (r : K) : GuaranteedAt q p₀ r 0 ↔ r ≤ ∑ θ, p₀ θ * q θ := by
  constructor
  · intro h
    exact h p₀ ⟨hp0, hp1, by rw [totalVariation_self]⟩
  · intro h p ⟨_, _, htv⟩
    have htv0 : totalVariation p p₀ = 0 := le_antisymm htv (totalVariation_nonneg p p₀)
    have hpeq : p = p₀ := (totalVariation_eq_zero_iff p p₀).mp htv0
    rw [hpeq]
    exact h

/-- At horizon `α = 1`, robustness holds iff the worst-state payoff meets `r`. -/
theorem guaranteedAt_one_iff {q p₀ : Θ → K} (hp0 : NonnegPrior p₀) (hp1 : NormalisedPrior p₀)
    (r : K) : GuaranteedAt q p₀ r 1 ↔ r ≤ minOver q := by
  classical
  constructor
  · intro h
    obtain ⟨θmin, hmin⟩ := exists_eq_minOver q
    let d : Θ → K := fun θ => if θ = θmin then 1 else 0
    have hd0 : NonnegPrior d := fun θ => by by_cases hθ : θ = θmin <;> simp [d, hθ]
    have hd1 : NormalisedPrior d := by simp [NormalisedPrior, d]
    have htv : totalVariation d p₀ ≤ 1 := totalVariation_le_one hd0 hd1 hp0 hp1
    have hval := h d ⟨hd0, hd1, htv⟩
    have hsum : ∑ θ, d θ * q θ = q θmin := by simp [d]
    rw [hsum, ← hmin] at hval
    exact hval
  · intro h p ⟨hpn, hp1', _⟩
    calc r ≤ minOver q := h
      _ = ∑ θ, p θ * minOver q := by rw [← Finset.sum_mul, hp1', one_mul]
      _ ≤ ∑ θ, p θ * q θ :=
        Finset.sum_le_sum fun θ _ => mul_le_mul_of_nonneg_left (minOver_le q θ) (hpn θ)

/-- At horizon `α = 0`, opportuneness holds iff the nominal expected payoff meets `w`. -/
theorem achievableAt_zero_iff {q p₀ : Θ → K} (hp0 : NonnegPrior p₀) (hp1 : NormalisedPrior p₀)
    (w : K) : AchievableAt q p₀ w 0 ↔ w ≤ ∑ θ, p₀ θ * q θ := by
  constructor
  · rintro ⟨p, ⟨_, _, htv⟩, hval⟩
    have hpeq : p = p₀ := (totalVariation_eq_zero_iff p p₀).mp
      (le_antisymm htv (totalVariation_nonneg p p₀))
    rwa [hpeq] at hval
  · intro h
    exact ⟨p₀, ⟨hp0, hp1, by rw [totalVariation_self]⟩, h⟩

/-- At horizon `α = 1`, opportuneness holds iff the best-state payoff meets `w`. -/
theorem achievableAt_one_iff {q p₀ : Θ → K} (hp0 : NonnegPrior p₀) (hp1 : NormalisedPrior p₀)
    (w : K) : AchievableAt q p₀ w 1 ↔ w ≤ maxOver q := by
  classical
  constructor
  · rintro ⟨p, ⟨hpn, hp1', _⟩, hval⟩
    calc w ≤ ∑ θ, p θ * q θ := hval
      _ ≤ ∑ θ, p θ * maxOver q :=
        Finset.sum_le_sum fun θ _ => mul_le_mul_of_nonneg_left (le_maxOver q θ) (hpn θ)
      _ = maxOver q := by rw [← Finset.sum_mul, hp1', one_mul]
  · intro h
    obtain ⟨θmax, hmax⟩ := exists_eq_maxOver q
    let d : Θ → K := fun θ => if θ = θmax then 1 else 0
    have hd0 : NonnegPrior d := fun θ => by by_cases hθ : θ = θmax <;> simp [d, hθ]
    have hd1 : NormalisedPrior d := by simp [NormalisedPrior, d]
    have htv : totalVariation d p₀ ≤ 1 := totalVariation_le_one hd0 hd1 hp0 hp1
    refine ⟨d, ⟨hd0, hd1, htv⟩, ?_⟩
    have hsum : ∑ θ, d θ * q θ = q θmax := by simp [d]
    rw [hsum, ← hmax]
    exact h

/-! ### Exact mass-shift equivalence over the total-variation ball -/

private theorem sum_pos_part_eq_tv {p p₀ : Θ → K} (hp1 : NormalisedPrior p)
    (hp01 : NormalisedPrior p₀) :
    ∑ θ, max 0 (p₀ θ - p θ) = totalVariation p p₀ := by
  have h1 : ∀ θ, 2 * max 0 (p₀ θ - p θ) = (p₀ θ - p θ) + |p θ - p₀ θ| := by
    intro θ
    rcases le_total (p θ) (p₀ θ) with h | h
    · rw [max_eq_right (sub_nonneg.mpr h), abs_of_nonpos (sub_nonpos.mpr h)]; ring
    · rw [max_eq_left (sub_nonpos.mpr h), abs_of_nonneg (sub_nonneg.mpr h)]; ring
  have hsum : 2 * ∑ θ, max 0 (p₀ θ - p θ) = ∑ θ, (p₀ θ - p θ) + ∑ θ, |p θ - p₀ θ| := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun θ _ => h1 θ
  have hdiff : ∑ θ, (p₀ θ - p θ) = 0 := by
    rw [Finset.sum_sub_distrib, hp01, hp1, sub_self]
  unfold totalVariation
  linarith

/-- Every belief in `InTVBall p₀ α` reduces expected payoff by at most the drop of the feasible
shift `δ θ = max 0 (p₀ θ - p θ)`. -/
theorem tvBall_drop_le (q : Θ → K) {p₀ p : Θ → K} {α : K}
    (hp0 : NonnegPrior p₀) (hp01 : NormalisedPrior p₀) (hp : InTVBall p₀ α p) :
    ∃ δ : Θ → K, ShiftFeasible p₀ α δ ∧
      ∑ θ, p₀ θ * q θ - ∑ θ, p θ * q θ ≤ ∑ θ, δ θ * (q θ - minOver q) := by
  let δ : Θ → K := fun θ => max 0 (p₀ θ - p θ)
  have hδ0 : ∀ θ, 0 ≤ δ θ := fun θ => le_max_left _ _
  have hδp : ∀ θ, δ θ ≤ p₀ θ := fun θ => max_le (hp0 θ) (by linarith [hp.1 θ])
  have hδsum : ∑ θ, δ θ ≤ α := (sum_pos_part_eq_tv hp.2.1 hp01).trans_le hp.2.2
  refine ⟨δ, ⟨hδ0, hδp, hδsum⟩, ?_⟩
  have hdiff0 : ∑ θ, (p₀ θ - p θ) = 0 := by
    rw [Finset.sum_sub_distrib, hp01, hp.2.1, sub_self]
  have hrewrite : ∑ θ, p₀ θ * q θ - ∑ θ, p θ * q θ =
      ∑ θ, (p₀ θ - p θ) * (q θ - minOver q) := by
    have h1 : ∑ θ, (p₀ θ - p θ) * (q θ - minOver q) =
        (∑ θ, p₀ θ * q θ - ∑ θ, p θ * q θ) - (∑ θ, (p₀ θ - p θ)) * minOver q := by
      rw [← Finset.sum_sub_distrib, Finset.sum_mul, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun θ _ => by ring
    rw [h1, hdiff0, zero_mul, sub_zero]
  rw [hrewrite]
  exact Finset.sum_le_sum fun θ _ =>
    mul_le_mul_of_nonneg_right (le_max_right 0 (p₀ θ - p θ)) (sub_nonneg.mpr (minOver_le q θ))

/-- Every feasible mass shift `δ` is attained by a belief in `InTVBall p₀ α` (moving mass `δ`
onto a worst-payoff state). -/
theorem exists_tvBall_attaining_shift (q : Θ → K) {p₀ δ : Θ → K} {α : K}
    (hp01 : NormalisedPrior p₀) (hδ : ShiftFeasible p₀ α δ) :
    ∃ p : Θ → K, InTVBall p₀ α p ∧
      ∑ θ, p₀ θ * q θ - ∑ θ, p θ * q θ = ∑ θ, δ θ * (q θ - minOver q) := by
  classical
  obtain ⟨θmin, hmin⟩ := exists_eq_minOver q
  let s : K := ∑ θ', δ θ'
  have hs0 : 0 ≤ s := Finset.sum_nonneg fun θ _ => hδ.1 θ
  let p : Θ → K := fun θ => p₀ θ - δ θ + if θ = θmin then s else 0
  have hpnn : NonnegPrior p := fun θ => by
    have h1 : 0 ≤ p₀ θ - δ θ := sub_nonneg.mpr (hδ.2.1 θ)
    have h2 : 0 ≤ (if θ = θmin then s else 0) := by split_ifs <;> [exact hs0; exact le_refl 0]
    exact add_nonneg h1 h2
  have hp1 : NormalisedPrior p := by
    change ∑ θ, (p₀ θ - δ θ + if θ = θmin then s else 0) = 1
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, hp01, Finset.sum_ite_eq']
    simp [s]
  have htv : totalVariation p p₀ ≤ α := by
    rw [← sum_pos_part_eq_tv hp1 hp01]
    refine le_trans (Finset.sum_le_sum fun θ _ => ?_) hδ.2.2
    have hle : p₀ θ - p θ ≤ δ θ := by
      have h2 : 0 ≤ (if θ = θmin then s else 0) := by split_ifs <;> [exact hs0; exact le_refl 0]
      change p₀ θ - (p₀ θ - δ θ + if θ = θmin then s else 0) ≤ δ θ
      linarith
    exact max_le (hδ.1 θ) hle
  refine ⟨p, ⟨hpnn, hp1, htv⟩, ?_⟩
  have hdrop : ∑ θ, p₀ θ * q θ - ∑ θ, p θ * q θ =
      ∑ θ, δ θ * q θ - s * q θmin := by
    have hpq : ∑ θ, p θ * q θ = ∑ θ, p₀ θ * q θ - ∑ θ, δ θ * q θ + s * q θmin := by
      change ∑ θ, (p₀ θ - δ θ + if θ = θmin then s else 0) * q θ = _
      simp only [add_mul, sub_mul, Finset.sum_add_distrib, Finset.sum_sub_distrib]
      congr 1
      have hite : ∀ θ, (if θ = θmin then s else 0) * q θ = if θ = θmin then s * q θmin else 0 := by
        intro θ
        by_cases h : θ = θmin <;> simp [h]
      simp only [hite, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    linarith
  rw [hdrop, ← hmin]
  change ∑ θ, δ θ * q θ - (∑ θ, δ θ) * minOver q = ∑ θ, δ θ * (q θ - minOver q)
  rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun θ _ => by ring

/-- Every belief in `InTVBall p₀ α` increases expected payoff by at most the gain of the feasible
shift `δ θ = max 0 (p₀ θ - p θ)`. -/
theorem tvBall_gain_le (q : Θ → K) {p₀ p : Θ → K} {α : K}
    (hp0 : NonnegPrior p₀) (hp01 : NormalisedPrior p₀) (hp : InTVBall p₀ α p) :
    ∃ δ : Θ → K, ShiftFeasible p₀ α δ ∧
      ∑ θ, p θ * q θ - ∑ θ, p₀ θ * q θ ≤ ∑ θ, δ θ * (maxOver q - q θ) := by
  let δ : Θ → K := fun θ => max 0 (p₀ θ - p θ)
  have hδ0 : ∀ θ, 0 ≤ δ θ := fun θ => le_max_left _ _
  have hδp : ∀ θ, δ θ ≤ p₀ θ := fun θ => max_le (hp0 θ) (by linarith [hp.1 θ])
  have hδsum : ∑ θ, δ θ ≤ α := (sum_pos_part_eq_tv hp.2.1 hp01).trans_le hp.2.2
  refine ⟨δ, ⟨hδ0, hδp, hδsum⟩, ?_⟩
  have hdiff0 : ∑ θ, (p₀ θ - p θ) = 0 := by
    rw [Finset.sum_sub_distrib, hp01, hp.2.1, sub_self]
  have hrewrite : ∑ θ, p θ * q θ - ∑ θ, p₀ θ * q θ =
      ∑ θ, (p₀ θ - p θ) * (maxOver q - q θ) := by
    have h1 : ∑ θ, (p₀ θ - p θ) * (maxOver q - q θ) =
        (∑ θ, (p₀ θ - p θ)) * maxOver q + (∑ θ, p θ * q θ - ∑ θ, p₀ θ * q θ) := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun θ _ => by ring
    rw [h1, hdiff0, zero_mul, zero_add]
  rw [hrewrite]
  exact Finset.sum_le_sum fun θ _ =>
    mul_le_mul_of_nonneg_right (le_max_right 0 (p₀ θ - p θ)) (sub_nonneg.mpr (le_maxOver q θ))

/-- Every feasible mass shift `δ` is attained by a belief in `InTVBall p₀ α` (moving mass `δ`
onto a best-payoff state). -/
theorem exists_tvBall_attaining_gain_shift (q : Θ → K) {p₀ δ : Θ → K} {α : K}
    (hp01 : NormalisedPrior p₀) (hδ : ShiftFeasible p₀ α δ) :
    ∃ p : Θ → K, InTVBall p₀ α p ∧
      ∑ θ, p θ * q θ - ∑ θ, p₀ θ * q θ = ∑ θ, δ θ * (maxOver q - q θ) := by
  classical
  obtain ⟨θmax, hmax⟩ := exists_eq_maxOver q
  let s : K := ∑ θ', δ θ'
  have hs0 : 0 ≤ s := Finset.sum_nonneg fun θ _ => hδ.1 θ
  let p : Θ → K := fun θ => p₀ θ - δ θ + if θ = θmax then s else 0
  have hpnn : NonnegPrior p := fun θ => by
    have h1 : 0 ≤ p₀ θ - δ θ := sub_nonneg.mpr (hδ.2.1 θ)
    have h2 : 0 ≤ (if θ = θmax then s else 0) := by split_ifs <;> [exact hs0; exact le_refl 0]
    exact add_nonneg h1 h2
  have hp1 : NormalisedPrior p := by
    change ∑ θ, (p₀ θ - δ θ + if θ = θmax then s else 0) = 1
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, hp01, Finset.sum_ite_eq']
    simp [s]
  have htv : totalVariation p p₀ ≤ α := by
    rw [← sum_pos_part_eq_tv hp1 hp01]
    refine le_trans (Finset.sum_le_sum fun θ _ => ?_) hδ.2.2
    have hle : p₀ θ - p θ ≤ δ θ := by
      have h2 : 0 ≤ (if θ = θmax then s else 0) := by split_ifs <;> [exact hs0; exact le_refl 0]
      change p₀ θ - (p₀ θ - δ θ + if θ = θmax then s else 0) ≤ δ θ
      linarith
    exact max_le (hδ.1 θ) hle
  refine ⟨p, ⟨hpnn, hp1, htv⟩, ?_⟩
  have hgain : ∑ θ, p θ * q θ - ∑ θ, p₀ θ * q θ =
      s * q θmax - ∑ θ, δ θ * q θ := by
    have hpq : ∑ θ, p θ * q θ = ∑ θ, p₀ θ * q θ - ∑ θ, δ θ * q θ + s * q θmax := by
      change ∑ θ, (p₀ θ - δ θ + if θ = θmax then s else 0) * q θ = _
      simp only [add_mul, sub_mul, Finset.sum_add_distrib, Finset.sum_sub_distrib]
      congr 1
      have hite : ∀ θ, (if θ = θmax then s else 0) * q θ = if θ = θmax then s * q θmax else 0 := by
        intro θ
        by_cases h : θ = θmax <;> simp [h]
      simp only [hite, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    linarith
  rw [hgain, ← hmax]
  change (∑ θ, δ θ) * maxOver q - ∑ θ, δ θ * q θ = ∑ θ, δ θ * (maxOver q - q θ)
  rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun θ _ => by ring

/-! ### Global optimality of greedy threshold-saturating shifts -/

/-- Any threshold-saturating feasible shift `δ*` (exhausting every state with `q θ > t`, taking
nothing from states with `q θ < t`, and either exhausting budget `α` or reaching `t = minOver q`)
maximises the worst-case drop over all feasible shifts. Backs `_tv_ball_robustness`. -/
theorem greedy_drop_shift_optimal (q : Θ → K) {p₀ δStar δ : Θ → K} {α t : K}
    (ht : minOver q ≤ t)
    (hhigh : ∀ θ, t < q θ → δStar θ = p₀ θ)
    (hlow : ∀ θ, q θ < t → δStar θ = 0)
    (hslack : (t - minOver q) * (α - ∑ θ, δStar θ) = 0)
    (hδ : ShiftFeasible p₀ α δ) :
    ∑ θ, δ θ * (q θ - minOver q) ≤ ∑ θ, δStar θ * (q θ - minOver q) := by
  have hdecomp : ∑ θ, δStar θ * (q θ - minOver q) - ∑ θ, δ θ * (q θ - minOver q) =
      ∑ θ, (δStar θ - δ θ) * (q θ - t) +
        (t - minOver q) * (∑ θ, δStar θ - ∑ θ, δ θ) := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, Finset.mul_sum,
      ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun θ _ => by ring
  have h1 : 0 ≤ ∑ θ, (δStar θ - δ θ) * (q θ - t) := by
    refine Finset.sum_nonneg fun θ _ => ?_
    rcases lt_trichotomy (q θ) t with hlt | heq | hgt
    · rw [hlow θ hlt]
      exact mul_nonneg_of_nonpos_of_nonpos (by linarith [hδ.1 θ]) (sub_nonpos.mpr hlt.le)
    · rw [heq, sub_self, mul_zero]
    · rw [hhigh θ hgt]
      exact mul_nonneg (sub_nonneg.mpr (hδ.2.1 θ)) (sub_nonneg.mpr hgt.le)
  have h2 : 0 ≤ (t - minOver q) * (∑ θ, δStar θ - ∑ θ, δ θ) := by
    have hrewrite : (t - minOver q) * (∑ θ, δStar θ - ∑ θ, δ θ) =
        (t - minOver q) * (α - ∑ θ, δ θ) - (t - minOver q) * (α - ∑ θ, δStar θ) := by ring
    rw [hrewrite, hslack, sub_zero]
    exact mul_nonneg (sub_nonneg.mpr ht) (sub_nonneg.mpr hδ.2.2)
  linarith

/-- Any threshold-saturating feasible shift `δ*` (exhausting every state with `q θ < t`, taking
nothing from states with `q θ > t`, and either exhausting budget `α` or reaching `t = maxOver q`)
maximises the best-case gain over all feasible shifts. Backs `_tv_ball_opportuneness`. -/
theorem greedy_gain_shift_optimal (q : Θ → K) {p₀ δStar δ : Θ → K} {α t : K}
    (ht : t ≤ maxOver q)
    (hlow : ∀ θ, q θ < t → δStar θ = p₀ θ)
    (hhigh : ∀ θ, t < q θ → δStar θ = 0)
    (hslack : (maxOver q - t) * (α - ∑ θ, δStar θ) = 0)
    (hδ : ShiftFeasible p₀ α δ) :
    ∑ θ, δ θ * (maxOver q - q θ) ≤ ∑ θ, δStar θ * (maxOver q - q θ) := by
  have hdecomp : ∑ θ, δStar θ * (maxOver q - q θ) - ∑ θ, δ θ * (maxOver q - q θ) =
      ∑ θ, (δStar θ - δ θ) * (t - q θ) +
        (maxOver q - t) * (∑ θ, δStar θ - ∑ θ, δ θ) := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, Finset.mul_sum,
      ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun θ _ => by ring
  have h1 : 0 ≤ ∑ θ, (δStar θ - δ θ) * (t - q θ) := by
    refine Finset.sum_nonneg fun θ _ => ?_
    rcases lt_trichotomy (q θ) t with hlt | heq | hgt
    · rw [hlow θ hlt]
      exact mul_nonneg (sub_nonneg.mpr (hδ.2.1 θ)) (sub_nonneg.mpr hlt.le)
    · rw [heq, sub_self, mul_zero]
    · rw [hhigh θ hgt]
      exact mul_nonneg_of_nonpos_of_nonpos (by linarith [hδ.1 θ]) (sub_nonpos.mpr hgt.le)
  have h2 : 0 ≤ (maxOver q - t) * (∑ θ, δStar θ - ∑ θ, δ θ) := by
    have hrewrite : (maxOver q - t) * (∑ θ, δStar θ - ∑ θ, δ θ) =
        (maxOver q - t) * (α - ∑ θ, δ θ) - (maxOver q - t) * (α - ∑ θ, δStar θ) := by ring
    rw [hrewrite, hslack, sub_zero]
    exact mul_nonneg (sub_nonneg.mpr ht) (sub_nonneg.mpr hδ.2.2)
  linarith

/-- A threshold-optimal feasible shift certifies `GuaranteedAt` across the whole total-variation
ball. -/
theorem guaranteedAt_of_greedy_drop (q : Θ → K) {p₀ δStar : Θ → K} {r α t : K}
    (hp0 : NonnegPrior p₀) (hp01 : NormalisedPrior p₀)
    (ht : minOver q ≤ t)
    (hhigh : ∀ θ, t < q θ → δStar θ = p₀ θ)
    (hlow : ∀ θ, q θ < t → δStar θ = 0)
    (hslack : (t - minOver q) * (α - ∑ θ, δStar θ) = 0)
    (hr : r ≤ ∑ θ, p₀ θ * q θ - ∑ θ, δStar θ * (q θ - minOver q)) :
    GuaranteedAt q p₀ r α := by
  intro p hp
  obtain ⟨δ, hδ, hdrop⟩ := tvBall_drop_le q hp0 hp01 hp
  have hopt := greedy_drop_shift_optimal q ht hhigh hlow hslack hδ
  linarith

/-- A threshold-optimal feasible shift whose attained gain falls strictly short of `w - E_{p₀}[q]`
certifies `¬ AchievableAt` across the whole total-variation ball. -/
theorem not_achievableAt_of_greedy_gain (q : Θ → K) {p₀ δStar : Θ → K} {w α t : K}
    (hp0 : NonnegPrior p₀) (hp01 : NormalisedPrior p₀)
    (ht : t ≤ maxOver q)
    (hlow : ∀ θ, q θ < t → δStar θ = p₀ θ)
    (hhigh : ∀ θ, t < q θ → δStar θ = 0)
    (hslack : (maxOver q - t) * (α - ∑ θ, δStar θ) = 0)
    (hw : ∑ θ, p₀ θ * q θ + ∑ θ, δStar θ * (maxOver q - q θ) < w) :
    ¬ AchievableAt q p₀ w α := by
  rintro ⟨p, hp, hval⟩
  obtain ⟨δ, hδ, hgain⟩ := tvBall_gain_le q hp0 hp01 hp
  have hopt := greedy_gain_shift_optimal q ht hlow hhigh hslack hδ
  linarith

end DecisionAnalysisProofs
