import DecisionAnalysisProofs.Finite.Problem
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!
# DecisionAnalysisProofs.Finite.Fidelity

The decision-loss bound of `src/fidelity.jl`: `total_variation(p, q)` and
`decision_loss_bound(B, ε) = 2 B ε`.

## Conventions (they fix the constant)

* `totalVariation p q` is **half** the ℓ¹ distance, `(Σ_θ |p θ - q θ|) / 2`, exactly the value
  returned by `total_variation`.
* The utility range `B` enters as a hypothesis, `∀ a θ θ', u a θ - u a θ' ≤ B`: within each
  **row** (action) of the payoff table, payoffs differ by at most `B`. The range of the whole
  table (`maximum(Q) - minimum(Q)`, the `B` of the Julia docstring and of the property test)
  satisfies it; the hypothesis is weaker, so the Julia bound also holds with `B` the largest
  row range.
* The two beliefs need only have the **same total mass** (`Σ p = Σ q`); nonnegativity is not
  used.

## What each result backs

* **`abs_expectedUtility_sub_le`** backs the docstring of `decision_loss_bound`: "for every
  action `|E_p u(a) − E_q u(a)| ≤ B ε`".
* **T10**, **`IsBayes.decision_loss_le`** backs `decision_loss_bound(B, ε)` and the property
  test in `test/test_voi.jl` ("The 2Bε bound: deciding under a wrong belief q loses at most
  2 B TV(p, q)"): an action that is Bayes under `q` falls short of the best under `p` by at
  most `2 · B · totalVariation p q`. **`decision_loss_nonneg`** is the `0 <= loss` half of the
  same test. **`decision_loss_le`** is the form with an `ε'`-optimal choice
  (`≤ 2 B ε + ε'`), which is what applies when `decide` is run with a nonzero tolerance.
* **`decision_loss_bound_tight`** backs the docstring's claim that the constant is `2 B` for
  these conventions: an instance with `B = 1`, `ε = 1/4` and loss exactly `1/2`.
* **`totalVariation_le_one`**, **`totalVariation_nonneg`** back "the result lies in `[0, 1]`"
  in the docstring of `total_variation`; **`totalVariation_eq_zero_iff`** is used by
  `Finite/Deficiency.lean` (`deficiency == 0` exactly for a garbling).

Not claimed: the identity `TV(p, q) = max_A |p(A) − q(A)|` quoted in the docstring of
`total_variation`; anything about floating-point evaluation of the bound; and no bound for
kernels or composite models (T11 in `Roadmap.lean`; the package has no function for it).
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ : Type} [Fintype A] [Nonempty A] [Fintype Θ]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- `total_variation(p, q)`: half the ℓ¹ distance. -/
def totalVariation (p q : Θ → K) : K := (∑ θ, |p θ - q θ|) / 2

theorem totalVariation_nonneg (p q : Θ → K) : 0 ≤ totalVariation p q :=
  div_nonneg (Finset.sum_nonneg fun _ _ => abs_nonneg _) zero_le_two

theorem totalVariation_comm (p q : Θ → K) : totalVariation p q = totalVariation q p := by
  unfold totalVariation
  exact congrArg (· / 2) (Finset.sum_congr rfl fun θ _ => abs_sub_comm _ _)

theorem totalVariation_self (p : Θ → K) : totalVariation p p = 0 := by
  simp [totalVariation]

/-- Total variation is zero exactly for equal vectors. -/
theorem totalVariation_eq_zero_iff (p q : Θ → K) : totalVariation p q = 0 ↔ p = q := by
  constructor
  · intro h
    have hsum : ∑ θ, |p θ - q θ| = 0 := by
      unfold totalVariation at h
      linarith
    funext θ
    have hθ := (Finset.sum_eq_zero_iff_of_nonneg fun θ _ => abs_nonneg (p θ - q θ)).mp hsum θ
      (Finset.mem_univ θ)
    exact sub_eq_zero.mp (abs_eq_zero.mp hθ)
  · rintro rfl
    exact totalVariation_self p

/-- For probability vectors the total variation is at most one. -/
theorem totalVariation_le_one {p q : Θ → K} (hp : NonnegPrior p) (hp1 : NormalisedPrior p)
    (hq : NonnegPrior q) (hq1 : NormalisedPrior q) : totalVariation p q ≤ 1 := by
  have h : ∑ θ, |p θ - q θ| ≤ ∑ θ, (p θ + q θ) :=
    Finset.sum_le_sum fun θ _ => by
      have h1 := hp θ
      have h2 := hq θ
      exact abs_le.mpr ⟨by linarith, by linarith⟩
  rw [Finset.sum_add_distrib] at h
  unfold NormalisedPrior at hp1 hq1
  unfold totalVariation
  linarith

/-- The inequality behind every bound of this file: against two weightings of equal total
mass, a function whose values differ by at most `B` has expectations differing by at most
`B` times the total variation. -/
theorem sum_sub_mul_le (p q : Θ → K) (hpq : ∑ θ, p θ = ∑ θ, q θ) (f : Θ → K) {B : K}
    (hB : ∀ θ θ', f θ - f θ' ≤ B) :
    ∑ θ, (p θ - q θ) * f θ ≤ B * totalVariation p q := by
  rcases isEmpty_or_nonempty Θ with hΘ | hΘ
  · simp [totalVariation]
  · obtain ⟨i, hi⟩ := exists_eq_maxOver f
    obtain ⟨j, hj⟩ := exists_eq_minOver f
    obtain ⟨c, hc⟩ : ∃ c : K, c = (maxOver f + minOver f) / 2 := ⟨_, rfl⟩
    have hosc : maxOver f - minOver f ≤ B := by
      rw [hi, hj]
      exact hB i j
    have hbound : ∀ θ, |f θ - c| ≤ B / 2 := by
      intro θ
      have h1 := le_maxOver f θ
      have h2 := minOver_le f θ
      exact abs_le.mpr ⟨by linarith, by linarith⟩
    have hzero : ∑ θ, (p θ - q θ) * c = 0 := by
      rw [← Finset.sum_mul, Finset.sum_sub_distrib, hpq, sub_self, zero_mul]
    have hshift : ∑ θ, (p θ - q θ) * f θ = ∑ θ, (p θ - q θ) * (f θ - c) := by
      simp only [mul_sub, Finset.sum_sub_distrib, hzero, sub_zero]
    rw [hshift]
    calc ∑ θ, (p θ - q θ) * (f θ - c)
        ≤ ∑ θ, |p θ - q θ| * (B / 2) :=
          Finset.sum_le_sum fun θ _ =>
            calc (p θ - q θ) * (f θ - c) ≤ |(p θ - q θ) * (f θ - c)| := le_abs_self _
              _ = |p θ - q θ| * |f θ - c| := abs_mul _ _
              _ ≤ |p θ - q θ| * (B / 2) :=
                mul_le_mul_of_nonneg_left (hbound θ) (abs_nonneg _)
      _ = B * totalVariation p q := by
          rw [← Finset.sum_mul, totalVariation]
          ring

omit [Nonempty A] in
/-- The expected utility of one action moves by at most `B · TV(p, q)` when the belief moves
from `q` to `p`, with `B` the range of that action's payoffs over states. -/
theorem abs_expectedUtility_sub_le (u : A → Θ → K) {p q : Θ → K}
    (hpq : ∑ θ, p θ = ∑ θ, q θ) (a : A) {B : K} (hB : ∀ θ θ', u a θ - u a θ' ≤ B) :
    |expectedUtility u p a - expectedUtility u q a| ≤ B * totalVariation p q := by
  have key : ∀ p q : Θ → K, expectedUtility u p a - expectedUtility u q a =
      ∑ θ, (p θ - q θ) * u a θ := fun p q => by
    simp [expectedUtility, sub_mul, Finset.sum_sub_distrib]
  have h1 := sum_sub_mul_le p q hpq (u a) hB
  have h2 := sum_sub_mul_le q p hpq.symm (u a) hB
  rw [← key] at h1 h2
  rw [totalVariation_comm q p] at h2
  exact abs_le.mpr ⟨by linarith, h1⟩

/-- The loss from any action is nonnegative (the `0 <= loss` of the Julia property test). -/
theorem decision_loss_nonneg (u : A → Θ → K) (p : Θ → K) (a : A) :
    0 ≤ actNowValue u p - expectedUtility u p a :=
  sub_nonneg.mpr (expectedUtility_le_actNowValue u p a)

/-- **T10**, with slack. An action that is within `ε'` of the best under the belief `q` falls
short of the best under `p` by at most `2 · B · TV(p, q) + ε'`. -/
theorem decision_loss_le (u : A → Θ → K) {p q : Θ → K} (hpq : ∑ θ, p θ = ∑ θ, q θ) {B : K}
    (hB : ∀ a θ θ', u a θ - u a θ' ≤ B) {a : A} {ε' : K}
    (ha : ∀ a', expectedUtility u q a' ≤ expectedUtility u q a + ε') :
    actNowValue u p - expectedUtility u p a ≤ 2 * B * totalVariation p q + ε' := by
  obtain ⟨b, hb⟩ := exists_eq_maxOver (expectedUtility u p)
  have h1 := (abs_le.mp (abs_expectedUtility_sub_le u hpq b (hB b))).2
  have h2 := (abs_le.mp (abs_expectedUtility_sub_le u hpq a (hB a))).1
  have h3 := ha b
  unfold actNowValue
  rw [hb]
  linarith

/-- **T10** (`decision_loss_bound`). An action that is Bayes under `q` falls short of the best
under `p`, in expectation under `p`, by at most `2 · B · TV(p, q)`. -/
theorem IsBayes.decision_loss_le {u : A → Θ → K} {p q : Θ → K} {a : A} (h : IsBayes u q a)
    (hpq : ∑ θ, p θ = ∑ θ, q θ) {B : K} (hB : ∀ a θ θ', u a θ - u a θ' ≤ B) :
    actNowValue u p - expectedUtility u p a ≤ 2 * B * totalVariation p q := by
  have := DecisionAnalysisProofs.decision_loss_le u hpq hB (a := a) (ε' := 0)
    fun a' => by rw [add_zero]; exact h a'
  linarith

/-! ### The constant cannot be lowered -/

/-- Two actions, two states, payoff range `B = 1`. -/
def tightU : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 1 | 0, 1 => 0
  | 1, 0 => 0 | 1, 1 => 1

/-- The true belief. -/
def tightP : Fin 2 → ℚ
  | 0 => 3/4 | 1 => 1/4

/-- The belief used to decide; both actions are tied under it. -/
def tightQ : Fin 2 → ℚ
  | 0 => 1/2 | 1 => 1/2

/-- The bound `2 B ε` is attained: with range `B = 1` and `TV(p, q) = 1/4`, an action that is
Bayes under `q` loses exactly `1/2` under `p`. No constant below `2` is valid for these
conventions. -/
theorem decision_loss_bound_tight :
    (∀ a θ θ', tightU a θ - tightU a θ' ≤ 1) ∧ totalVariation tightP tightQ = 1/4 ∧
      IsBayes tightU tightQ 1 ∧
      actNowValue tightU tightP - expectedUtility tightU tightP 1 = 2 * 1 * (1/4) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro a θ θ'
    fin_cases a <;> fin_cases θ <;> fin_cases θ' <;> norm_num [tightU]
  · norm_num [totalVariation, Fin.sum_univ_two, tightP, tightQ, abs_of_nonneg, abs_of_nonpos]
  · intro a'
    fin_cases a' <;> norm_num [expectedUtility, Fin.sum_univ_two, tightU, tightQ]
  · norm_num [actNowValue, maxOver_fin_two, expectedUtility, Fin.sum_univ_two, tightU, tightP]

end DecisionAnalysisProofs
