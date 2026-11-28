import DecisionAnalysisProofs.Finite.Posterior
import DecisionAnalysisProofs.Finite.Credal
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.MeanInequalities

/-!
# DecisionAnalysisProofs.Finite.CertaintyEquivalentVOI

Certainty-equivalent value of information (`ce_evpi`, `ce_evppi`, `ce_evsi` in `src/ce_voi.jl`)
and smooth ambiguity aggregation (`SmoothAmbiguity` in `src/aggregators.jl`, tested in
`test/test_track3.jl` and `test/test_track4.jl`).

## Conventions

* **Linear certainty-equivalent VOI** (over any linearly ordered field `K`): under an increasing
  affine utility `x ↦ a * x + b` (`a > 0`), the certainty equivalent of expected utility `v` is
  `(v - b) / a`.
* **CARA certainty equivalent** (over `ℝ`, `r > 0`):
  `caraCE r p x = -(1 / r) * Real.log (∑ θ, p θ * Real.exp (-r * x θ))`.
  The associated utility is `x ↦ -Real.exp (-r * x)` (any positive affine rescaling, such as
  `(1 - exp(-r x)) / r`, yields the identical certainty equivalent and buying price).
* **KMM (2005) smooth ambiguity** (over `ℝ`, second-order prior `μ : I → ℝ` on credal generators
  `P : I → Θ → ℝ`):
  - general form `smoothAmbiguity φ φInv μ P x = φInv (∑ i, μ i * φ (∑ θ, P i θ * x θ))`,
  - exponential (constant absolute ambiguity aversion `η > 0`) form
    `expSmoothAmbiguity η μ P x = caraCE η μ (fun i => ∑ θ, P i θ * x θ)`.

## Headline results

* **`ceEvsi_linear_eq_evsi`**, **`ceEvpi_linear_eq_evpi`**, **`ceEvppi_linear_eq_evppi`**:
  under a linear utility `a * x + b` (`a > 0`), certainty-equivalent VOI equals plain outcome
  VOI identically (exact in rational arithmetic).
* **`caraCE_logSumExp_eq`**: shift-invariant log-sum-exp identity backing
  `_exp_certainty_equivalent`.
* **`caraCE_add_const`**, **`caraCE_const`**, **`caraCE_mono`**, **`minOver_le_caraCE`**,
  **`caraCE_le_maxOver`**, **`caraCE_le_mean`**, **`caraCE_concave`**: CARA certainty
  equivalents are translation-equivariant, monotone, bounded by `[min x, max x]`, bounded above
  by the mean (risk aversion via Jensen's inequality), and concave in the payoff vector.
* **`caraActNowCE_eq_maxOver_caraCE`**, **`cara_buying_price_eq_ce_gain`**,
  **`cara_buying_price_nonneg`**: Howard's (1966) buying-price theorem for CARA — the unique
  deduction `v*` solving `V_info(v*) = V_0` is the nonnegative closed form
  `caraInformedCE - caraActNowCE`, backing `_ce_exp_informed_finite`.
* **`expSmoothAmbiguity_logSumExp_eq`**, **`lowerExpectedUtility_le_expSmoothAmbiguity`**,
  **`expSmoothAmbiguity_le_mixture`**, **`expSmoothAmbiguity_le_upperExpectedUtility`**,
  **`expSmoothAmbiguity_mono`**, **`expSmoothAmbiguity_add_const`**,
  **`expSmoothAmbiguity_const`**, **`expSmoothAmbiguity_concave`**: exponential KMM smooth
  ambiguity (`η > 0`) matches its log-sum-exp implementation, sits between `LowerExpectation`
  and the `μ`-mixture expectation (hence below `UpperExpectation`), is monotone,
  translation-equivariant, constant-preserving, and concave (ambiguity-averse).
* **`smoothAmbiguity_le_mixture_of_concave`**: every smooth ambiguity aggregator with a
  strictly increasing concave `φ` lies at or below the `μ`-mixture expectation.

Not claimed: CRRA (`PowerUtility`) bisection tolerances or `OutcomeTable` replicate indexing.
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

/-! ### Linear certainty-equivalent VOI (any linearly ordered field `K`) -/

section LinearCE

variable {A Θ Z C : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z] [Fintype C]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The certainty equivalent of expected utility under linear utility `x ↦ a * x + b` (`a > 0`)
equals the expected outcome. -/
theorem ce_linear_eq_mean (p x : Θ → K) (hp1 : NormalisedPrior p) {a : K} (ha : a ≠ 0) (b : K) :
    ((∑ θ, p θ * (a * x θ + b)) - b) / a = ∑ θ, p θ * x θ := by
  have hsum : ∑ θ, p θ * (a * x θ + b) = a * ∑ θ, p θ * x θ + b := by
    have h1 : ∀ θ, p θ * (a * x θ + b) = a * (p θ * x θ) + p θ * b := fun θ => by ring
    rw [Finset.sum_congr rfl fun θ _ => h1 θ, Finset.sum_add_distrib, ← Finset.mul_sum,
      ← Finset.sum_mul, hp1, one_mul]
  rw [hsum, add_sub_cancel_right, mul_div_cancel_left₀ _ ha]

/-- Certainty-equivalent EVSI under linear utility `x ↦ a * x + b` (`a > 0`) equals plain EVSI. -/
theorem ceEvsi_linear_eq_evsi (x : A → Θ → K) {p : Θ → K} (hp1 : NormalisedPrior p)
    {e : Θ → Z → K} (he : NormalisedKernel e) {a : K} (ha : 0 < a) (b : K) :
    (valueWithInformation (fun act θ => a * x act θ + b) p e - b) / a -
      (actNowValue (fun act θ => a * x act θ + b) p - b) / a = evsi x p e := by
  have hres : (fun act θ => a * x act θ + b) = rescale a (fun _ => b) x := rfl
  rw [hres, valueWithInformation_rescale ha.le (fun _ => b) x p he,
    actNowValue_rescale ha.le (fun _ => b) x p]
  have hb : ∑ θ, p θ * b = b := by rw [← Finset.sum_mul, hp1, one_mul]
  rw [hb, add_sub_cancel_right, add_sub_cancel_right,
    mul_div_cancel_left₀ _ (ne_of_gt ha), mul_div_cancel_left₀ _ (ne_of_gt ha)]
  rfl

/-- Certainty-equivalent EVPI under linear utility `x ↦ a * x + b` (`a > 0`) equals plain EVPI. -/
theorem ceEvpi_linear_eq_evpi (x : A → Θ → K) {p : Θ → K} (hp1 : NormalisedPrior p)
    {a : K} (ha : 0 < a) (b : K) :
    (perfectValue (fun act θ => a * x act θ + b) p - b) / a -
      (actNowValue (fun act θ => a * x act θ + b) p - b) / a = evpi x p := by
  have hres : (fun act θ => a * x act θ + b) = rescale a (fun _ => b) x := rfl
  rw [hres, perfectValue_rescale ha.le (fun _ => b) x p,
    actNowValue_rescale ha.le (fun _ => b) x p]
  have hb : ∑ θ, p θ * b = b := by rw [← Finset.sum_mul, hp1, one_mul]
  rw [hb, add_sub_cancel_right, add_sub_cancel_right,
    mul_div_cancel_left₀ _ (ne_of_gt ha), mul_div_cancel_left₀ _ (ne_of_gt ha)]
  rfl

/-- Certainty-equivalent EVPPI under linear utility `x ↦ a * x + b` (`a > 0`) equals plain
EVPPI. -/
theorem ceEvppi_linear_eq_evppi [DecidableEq C] (x : A → Θ → K) {p : Θ → K}
    (hp1 : NormalisedPrior p) (cell : Θ → C) {a : K} (ha : 0 < a) (b : K) :
    (partialValue (fun act θ => a * x act θ + b) p cell - b) / a -
      (actNowValue (fun act θ => a * x act θ + b) p - b) / a = evppi x p cell := by
  have hpart : partialValue (fun act θ => a * x act θ + b) p cell =
      valueWithInformation (fun act θ => a * x act θ + b) p (deterministic cell) :=
    partialValue_eq_valueWithInformation _ p cell
  have hce := ceEvsi_linear_eq_evsi x hp1 (normalisedKernel_deterministic cell) ha b
  rw [hpart, hce, ← evppi_eq_evsi]

end LinearCE

/-! ### CARA certainty equivalent and Howard's buying-price theorem (over `ℝ`) -/

section CARA

variable {A Θ Z I : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Nonempty Θ]
  [Fintype Z] [Fintype I] [Nonempty I]

/-- The CARA certainty equivalent of lottery `(x, p)` at absolute risk aversion `r`:
`-(1 / r) * log (∑ θ, p θ * exp (-r * x θ))`. -/
noncomputable def caraCE (r : ℝ) (p x : Θ → ℝ) : ℝ :=
  -(1 / r) * Real.log (∑ θ, p θ * Real.exp (-r * x θ))

private theorem sum_prior_mul_exp_pos {p : Θ → ℝ} (hp : NonnegPrior p) (hp1 : NormalisedPrior p)
    (f : Θ → ℝ) : 0 < ∑ θ, p θ * Real.exp (f θ) := by
  obtain ⟨θ₀, hθ₀⟩ : ∃ θ₀, 0 < p θ₀ := by
    by_contra h
    have hle : ∑ θ, p θ ≤ 0 := Finset.sum_nonpos fun θ _ => not_lt.mp fun hpos => h ⟨θ, hpos⟩
    unfold NormalisedPrior at hp1
    linarith
  exact Finset.sum_pos' (fun θ _ => mul_nonneg (hp θ) (Real.exp_nonneg _))
    ⟨θ₀, Finset.mem_univ θ₀, mul_pos hθ₀ (Real.exp_pos _)⟩

/-- Shift-invariant log-sum-exp identity backing `_exp_certainty_equivalent`. -/
theorem caraCE_logSumExp_eq {r : ℝ} (hr : r ≠ 0) {p : Θ → ℝ} (hp : NonnegPrior p)
    (hp1 : NormalisedPrior p) (x : Θ → ℝ) (m : ℝ) :
    caraCE r p x = m - (1 / r) * Real.log (∑ θ, p θ * Real.exp (-r * (x θ - m))) := by
  unfold caraCE
  have hfactor : ∑ θ, p θ * Real.exp (-r * x θ) =
      Real.exp (-r * m) * ∑ θ, p θ * Real.exp (-r * (x θ - m)) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun θ _ => ?_
    rw [show p θ * Real.exp (-r * x θ) = Real.exp (-r * m) * (p θ * Real.exp (-r * (x θ - m))) by
      rw [mul_left_comm, ← Real.exp_add]; congr 2; ring]
  have hpos := sum_prior_mul_exp_pos hp hp1 (fun θ => -r * (x θ - m))
  rw [hfactor, Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt hpos), Real.log_exp]
  field_simp
  ring

/-- CARA certainty equivalent is translation-equivariant (constant absolute risk aversion). -/
theorem caraCE_add_const {r : ℝ} (hr : r ≠ 0) {p : Θ → ℝ} (hp : NonnegPrior p)
    (hp1 : NormalisedPrior p) (x : Θ → ℝ) (c : ℝ) :
    caraCE r p (fun θ => x θ + c) = caraCE r p x + c := by
  rw [caraCE_logSumExp_eq hr hp hp1 (fun θ => x θ + c) c]
  have hsum : ∑ θ, p θ * Real.exp (-r * (x θ + c - c)) = ∑ θ, p θ * Real.exp (-r * x θ) :=
    Finset.sum_congr rfl fun θ _ => by rw [add_sub_cancel_right]
  rw [hsum]
  unfold caraCE
  ring

/-- A constant lottery has certainty equivalent equal to that constant. -/
theorem caraCE_const {r : ℝ} (hr : r ≠ 0) {p : Θ → ℝ} (hp1 : NormalisedPrior p) (c : ℝ) :
    caraCE r p (fun _ => c) = c := by
  unfold caraCE
  rw [← Finset.sum_mul, hp1, one_mul, Real.log_exp]
  field_simp

/-- CARA certainty equivalent is monotone in the payoff vector for `r > 0`. -/
theorem caraCE_mono {r : ℝ} (hr : 0 < r) {p : Θ → ℝ} (hp : NonnegPrior p)
    (hp1 : NormalisedPrior p) {x y : Θ → ℝ} (hxy : ∀ θ, x θ ≤ y θ) :
    caraCE r p x ≤ caraCE r p y := by
  unfold caraCE
  have hsum_le : ∑ θ, p θ * Real.exp (-r * y θ) ≤ ∑ θ, p θ * Real.exp (-r * x θ) :=
    Finset.sum_le_sum fun θ _ =>
      mul_le_mul_of_nonneg_left
        (Real.exp_le_exp.mpr (by nlinarith [hxy θ])) (hp θ)
  have hpos_y := sum_prior_mul_exp_pos hp hp1 (fun θ => -r * y θ)
  have hlog : Real.log (∑ θ, p θ * Real.exp (-r * y θ)) ≤
      Real.log (∑ θ, p θ * Real.exp (-r * x θ)) :=
    Real.log_le_log hpos_y hsum_le
  have hneg : -(1 / r) ≤ 0 := by
    have : 0 < 1 / r := one_div_pos.mpr hr
    linarith
  exact mul_le_mul_of_nonpos_left hlog hneg

/-- CARA certainty equivalent is at least the minimum outcome. -/
theorem minOver_le_caraCE {r : ℝ} (hr : 0 < r) {p : Θ → ℝ} (hp : NonnegPrior p)
    (hp1 : NormalisedPrior p) (x : Θ → ℝ) :
    minOver x ≤ caraCE r p x := by
  rw [← caraCE_const (ne_of_gt hr) hp1 (minOver x)]
  exact caraCE_mono hr hp hp1 (minOver_le x)

/-- CARA certainty equivalent is at most the maximum outcome. -/
theorem caraCE_le_maxOver {r : ℝ} (hr : 0 < r) {p : Θ → ℝ} (hp : NonnegPrior p)
    (hp1 : NormalisedPrior p) (x : Θ → ℝ) :
    caraCE r p x ≤ maxOver x := by
  rw [← caraCE_const (ne_of_gt hr) hp1 (maxOver x)]
  exact caraCE_mono hr hp hp1 (le_maxOver x)

/-- Risk aversion: for `r > 0`, the CARA certainty equivalent never exceeds the expected payoff
(Jensen's inequality for `exp`). -/
theorem caraCE_le_mean {r : ℝ} (hr : 0 < r) {p : Θ → ℝ} (hp : NonnegPrior p)
    (hp1 : NormalisedPrior p) (x : Θ → ℝ) :
    caraCE r p x ≤ ∑ θ, p θ * x θ := by
  have hjensen : Real.exp (∑ θ, p θ * (-r * x θ)) ≤ ∑ θ, p θ * Real.exp (-r * x θ) :=
    convexOn_exp.map_sum_le (fun θ _ => hp θ) hp1 (fun _ _ => Set.mem_univ _)
  have hlog : ∑ θ, p θ * (-r * x θ) ≤ Real.log (∑ θ, p θ * Real.exp (-r * x θ)) := by
    calc ∑ θ, p θ * (-r * x θ)
        = Real.log (Real.exp (∑ θ, p θ * (-r * x θ))) := (Real.log_exp _).symm
      _ ≤ Real.log (∑ θ, p θ * Real.exp (-r * x θ)) :=
          Real.log_le_log (Real.exp_pos _) hjensen
  have hmul : ∑ θ, p θ * (-r * x θ) = -r * ∑ θ, p θ * x θ := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun θ _ => by ring
  rw [hmul] at hlog
  unfold caraCE
  have hinv : 0 < 1 / r := one_div_pos.mpr hr
  have hstep := mul_le_mul_of_nonneg_left hlog hinv.le
  have hlhs : 1 / r * (-r * ∑ θ, p θ * x θ) = -(∑ θ, p θ * x θ) := by
    field_simp
  rw [hlhs] at hstep
  linarith

/-- Concavity of the CARA certainty equivalent in the payoff vector (`r > 0`), proved via weighted
AM-GM (Hölder inequality for log-sum-exp). -/
theorem caraCE_concave {r : ℝ} (hr : 0 < r) {p : Θ → ℝ} (hp : NonnegPrior p)
    (hp1 : NormalisedPrior p) (x y : Θ → ℝ) {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    (1 - t) * caraCE r p x + t * caraCE r p y ≤
      caraCE r p (fun θ => (1 - t) * x θ + t * y θ) := by
  let A : ℝ := ∑ θ, p θ * Real.exp (-r * x θ)
  let B : ℝ := ∑ θ, p θ * Real.exp (-r * y θ)
  let C : ℝ := ∑ θ, p θ * Real.exp (-r * ((1 - t) * x θ + t * y θ))
  have hA : 0 < A := sum_prior_mul_exp_pos hp hp1 (fun θ => -r * x θ)
  have hB : 0 < B := sum_prior_mul_exp_pos hp hp1 (fun θ => -r * y θ)
  have hC : 0 < C := sum_prior_mul_exp_pos hp hp1 (fun θ => -r * ((1 - t) * x θ + t * y θ))
  have hterm : ∀ θ,
      (Real.exp (-r * x θ) / A) ^ (1 - t) * (Real.exp (-r * y θ) / B) ^ t ≤
        (1 - t) * (Real.exp (-r * x θ) / A) + t * (Real.exp (-r * y θ) / B) := fun θ =>
    Real.geom_mean_le_arith_mean2_weighted (sub_nonneg.mpr ht1) ht0
      (div_nonneg (Real.exp_nonneg _) hA.le) (div_nonneg (Real.exp_nonneg _) hB.le) (by ring)
  have hpow : ∀ θ,
      (Real.exp (-r * x θ) / A) ^ (1 - t) * (Real.exp (-r * y θ) / B) ^ t =
        Real.exp (-r * ((1 - t) * x θ + t * y θ)) / (A ^ (1 - t) * B ^ t) := by
    intro θ
    rw [Real.div_rpow (Real.exp_nonneg _) hA.le, Real.div_rpow (Real.exp_nonneg _) hB.le,
      div_mul_div_comm, ← Real.exp_mul, ← Real.exp_mul, ← Real.exp_add]
    congr 2
    ring
  have hsum : C / (A ^ (1 - t) * B ^ t) ≤ 1 := by
    calc C / (A ^ (1 - t) * B ^ t)
        = ∑ θ, p θ * (Real.exp (-r * ((1 - t) * x θ + t * y θ)) / (A ^ (1 - t) * B ^ t)) := by
          rw [Finset.sum_div]
          exact Finset.sum_congr rfl fun θ _ => by rw [mul_div_assoc]
      _ = ∑ θ, p θ * ((Real.exp (-r * x θ) / A) ^ (1 - t) * (Real.exp (-r * y θ) / B) ^ t) :=
          Finset.sum_congr rfl fun θ _ => by rw [hpow θ]
      _ ≤ ∑ θ, p θ * ((1 - t) * (Real.exp (-r * x θ) / A) +
            t * (Real.exp (-r * y θ) / B)) :=
          Finset.sum_le_sum fun θ _ => mul_le_mul_of_nonneg_left (hterm θ) (hp θ)
      _ = (1 - t) * (A / A) + t * (B / B) := by
          have h1 : ∀ θ, p θ * ((1 - t) * (Real.exp (-r * x θ) / A) +
              t * (Real.exp (-r * y θ) / B)) =
              (1 - t) * (p θ * Real.exp (-r * x θ) / A) +
              t * (p θ * Real.exp (-r * y θ) / B) := fun θ => by ring
          rw [Finset.sum_congr rfl fun θ _ => h1 θ, Finset.sum_add_distrib,
            ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.sum_div, ← Finset.sum_div]
      _ = 1 := by rw [div_self (ne_of_gt hA), div_self (ne_of_gt hB)]; ring
  have hAB : 0 < A ^ (1 - t) * B ^ t := mul_pos (Real.rpow_pos_of_pos hA _) (Real.rpow_pos_of_pos hB _)
  have hCle : C ≤ A ^ (1 - t) * B ^ t := (div_le_one hAB).mp hsum
  have hlog : Real.log C ≤ (1 - t) * Real.log A + t * Real.log B := by
    calc Real.log C ≤ Real.log (A ^ (1 - t) * B ^ t) := Real.log_le_log hC hCle
      _ = Real.log (A ^ (1 - t)) + Real.log (B ^ t) :=
          Real.log_mul (ne_of_gt (Real.rpow_pos_of_pos hA _)) (ne_of_gt (Real.rpow_pos_of_pos hB _))
      _ = (1 - t) * Real.log A + t * Real.log B := by
          rw [Real.log_rpow hA, Real.log_rpow hB]
  change (1 - t) * (-(1 / r) * Real.log A) + t * (-(1 / r) * Real.log B) ≤ -(1 / r) * Real.log C
  have hinv : 0 < 1 / r := one_div_pos.mpr hr
  nlinarith

/-! ### Howard's buying-price theorem under CARA -/

/-- CARA utility payoff table `u a θ = -exp(-r * x a θ)`. -/
noncomputable def caraTable (r : ℝ) (x : A → Θ → ℝ) : A → Θ → ℝ :=
  fun a θ => -Real.exp (-r * x a θ)

/-- Act-now CARA expected utility `V_0 = max_a ∑ θ, p θ * (-exp(-r * x a θ))`. -/
noncomputable def caraActNowVal (r : ℝ) (x : A → Θ → ℝ) (p : Θ → ℝ) : ℝ :=
  actNowValue (caraTable r x) p

/-- Informed CARA expected utility when paying buying price `v`:
`V_info(v) = ∑ z, max_a ∑ θ, p θ * e θ z * (-exp(-r * (x a θ - v)))`. -/
noncomputable def caraInformedVal (r : ℝ) (x : A → Θ → ℝ) (p : Θ → ℝ) (e : Θ → Z → ℝ)
    (v : ℝ) : ℝ :=
  valueWithInformation (caraTable r (fun a θ => x a θ - v)) p e

/-- Act-now CARA certainty equivalent `CE_0 = -(1/r) log(-V_0)`. -/
noncomputable def caraActNowCE (r : ℝ) (x : A → Θ → ℝ) (p : Θ → ℝ) : ℝ :=
  -(1 / r) * Real.log (-caraActNowVal r x p)

/-- Informed CARA certainty equivalent `CE_info = -(1/r) log(-V_info(0))`. -/
noncomputable def caraInformedCE (r : ℝ) (x : A → Θ → ℝ) (p : Θ → ℝ) (e : Θ → Z → ℝ) : ℝ :=
  -(1 / r) * Real.log (-caraInformedVal r x p e 0)

private theorem caraTable_shift (r : ℝ) (x : A → Θ → ℝ) (v : ℝ) :
    caraTable r (fun a θ => x a θ - v) = rescale (Real.exp (r * v)) (fun _ => 0) (caraTable r x) := by
  funext a θ
  unfold caraTable rescale
  rw [add_zero, show -r * (x a θ - v) = r * v + -r * x a θ by ring, Real.exp_add]
  ring

/-- Paying a state-independent deduction `v` multiplies informed CARA expected utility by
`exp(r * v)`. -/
theorem caraInformedVal_eq_exp_mul (r : ℝ) (x : A → Θ → ℝ) (p : Θ → ℝ) {e : Θ → Z → ℝ}
    (he : NormalisedKernel e) (v : ℝ) :
    caraInformedVal r x p e v = Real.exp (r * v) * caraInformedVal r x p e 0 := by
  unfold caraInformedVal
  have h0 : (fun a θ => x a θ - 0) = x := funext fun a => funext fun θ => sub_zero _
  rw [h0, caraTable_shift,
    valueWithInformation_rescale (Real.exp_pos (r * v)).le (fun _ => 0) (caraTable r x) p he]
  simp

private theorem caraActNowVal_neg (r : ℝ) (x : A → Θ → ℝ) {p : Θ → ℝ}
    (hp : NonnegPrior p) (hp1 : NormalisedPrior p) :
    caraActNowVal r x p < 0 := by
  obtain ⟨a, ha⟩ := exists_eq_maxOver (expectedUtility (caraTable r x) p)
  change maxOver (expectedUtility (caraTable r x) p) < 0
  rw [ha]
  unfold expectedUtility caraTable
  have hneg : ∑ θ, p θ * -Real.exp (-r * x a θ) = -(∑ θ, p θ * Real.exp (-r * x a θ)) := by
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun θ _ => by ring
  rw [hneg]
  linarith [sum_prior_mul_exp_pos hp hp1 (fun θ => -r * x a θ)]

private theorem caraInformedVal_zero_neg (r : ℝ) (x : A → Θ → ℝ) {p : Θ → ℝ}
    (hp : NonnegPrior p) (hp1 : NormalisedPrior p) {e : Θ → Z → ℝ}
    (he : StochasticKernel e) :
    caraInformedVal r x p e 0 < 0 := by
  classical
  have h0 : (fun a θ => x a θ - 0) = x := funext fun a => funext fun θ => sub_zero _
  unfold caraInformedVal
  rw [h0]
  have hle : valueWithInformation (caraTable r x) p e ≤ perfectValue (caraTable r x) p := by
    have h := evsi_le_evpi (caraTable r x) hp he
    unfold evsi evpi at h
    linarith
  have hperf : perfectValue (caraTable r x) p < 0 := by
    unfold perfectValue
    have hrow : ∀ θ, maxOver (fun a => caraTable r x a θ) < 0 := fun θ => by
      obtain ⟨a, ha⟩ := exists_eq_maxOver (fun a => caraTable r x a θ)
      rw [ha]
      exact neg_lt_zero.mpr (Real.exp_pos _)
    obtain ⟨θ₀, hθ₀⟩ : ∃ θ₀, 0 < p θ₀ := by
      by_contra h
      have : ∑ θ, p θ ≤ 0 := Finset.sum_nonpos fun θ _ => not_lt.mp fun hp0 => h ⟨θ, hp0⟩
      unfold NormalisedPrior at hp1
      linarith
    have hneg : 0 < ∑ θ, p θ * -maxOver (fun a => caraTable r x a θ) :=
      Finset.sum_pos' (fun θ _ => mul_nonneg (hp θ) (by linarith [hrow θ]))
        ⟨θ₀, Finset.mem_univ θ₀, mul_pos hθ₀ (by linarith [hrow θ₀])⟩
    have heq : ∑ θ, p θ * -maxOver (fun a => caraTable r x a θ) =
        -(∑ θ, p θ * maxOver (fun a => caraTable r x a θ)) := by
      rw [← Finset.sum_neg_distrib]
      exact Finset.sum_congr rfl fun θ _ => by ring
    linarith
  exact lt_of_le_of_lt hle hperf

/-- The act-now CARA certainty equivalent equals the maximum action-wise CARA certainty
equivalent. -/
theorem caraActNowCE_eq_maxOver_caraCE {r : ℝ} (hr : 0 < r) (x : A → Θ → ℝ) {p : Θ → ℝ}
    (hp : NonnegPrior p) (hp1 : NormalisedPrior p) :
    caraActNowCE r x p = maxOver (fun a => caraCE r p (x a)) := by
  obtain ⟨a₀, ha₀⟩ := exists_eq_maxOver (expectedUtility (caraTable r x) p)
  have hrow : ∀ a, -expectedUtility (caraTable r x) p a = ∑ θ, p θ * Real.exp (-r * x a θ) := by
    intro a
    unfold expectedUtility caraTable
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun θ _ => by ring
  have hval : caraActNowCE r x p = caraCE r p (x a₀) := by
    unfold caraActNowCE caraActNowVal actNowValue caraCE
    rw [ha₀, hrow a₀]
  symm
  refine maxOver_eq_of a₀ hval.symm fun a => ?_
  rw [hval]
  unfold caraCE
  have hle : expectedUtility (caraTable r x) p a ≤ expectedUtility (caraTable r x) p a₀ :=
    ha₀ ▸ le_maxOver (expectedUtility (caraTable r x) p) a
  have hsum : ∑ θ, p θ * Real.exp (-r * x a₀ θ) ≤ ∑ θ, p θ * Real.exp (-r * x a θ) := by
    rw [← hrow a₀, ← hrow a]
    linarith
  have hlog := Real.log_le_log (sum_prior_mul_exp_pos hp hp1 (fun θ => -r * x a₀ θ)) hsum
  have hneg : -(1 / r) ≤ 0 := by
    have : 0 < 1 / r := one_div_pos.mpr hr
    linarith
  exact mul_le_mul_of_nonpos_left hlog hneg

/-- **Howard's (1966) buying-price theorem for CARA**: for `r > 0`, the unique state-independent
deduction `v` satisfying `V_info(v) = V_0` is the certainty-equivalent gain
`v = caraInformedCE - caraActNowCE`. Backs `_ce_exp_informed_finite`. -/
theorem cara_buying_price_eq_ce_gain {r : ℝ} (hr : 0 < r) (x : A → Θ → ℝ) {p : Θ → ℝ}
    (hp : NonnegPrior p) (hp1 : NormalisedPrior p) {e : Θ → Z → ℝ}
    (he : StochasticKernel e) (v : ℝ) :
    caraInformedVal r x p e v = caraActNowVal r x p ↔
      v = caraInformedCE r x p e - caraActNowCE r x p := by
  have hact_pos : 0 < -caraActNowVal r x p := neg_pos.mpr (caraActNowVal_neg r x hp hp1)
  have hinf_pos : 0 < -caraInformedVal r x p e 0 :=
    neg_pos.mpr (caraInformedVal_zero_neg r x hp hp1 he)
  rw [caraInformedVal_eq_exp_mul r x p he.2 v]
  constructor
  · intro h
    have hneg : Real.exp (r * v) * (-caraInformedVal r x p e 0) = -caraActNowVal r x p := by
      linarith
    have hlog := congrArg Real.log hneg
    rw [Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt hinf_pos), Real.log_exp] at hlog
    unfold caraInformedCE caraActNowCE
    have hr0 : r ≠ 0 := ne_of_gt hr
    field_simp
    linarith
  · intro hv
    have hr0 : r ≠ 0 := ne_of_gt hr
    have hrv : r * v = Real.log (-caraActNowVal r x p) - Real.log (-caraInformedVal r x p e 0) := by
      rw [hv]
      unfold caraInformedCE caraActNowCE
      field_simp
      ring
    rw [hrv, Real.exp_sub, Real.exp_log hact_pos, Real.exp_log hinf_pos]
    have hne : caraInformedVal r x p e 0 ≠ 0 := by linarith
    field_simp

/-- The CARA certainty-equivalent VOI `caraInformedCE - caraActNowCE` is nonnegative. -/
theorem cara_buying_price_nonneg {r : ℝ} (hr : 0 < r) (x : A → Θ → ℝ) {p : Θ → ℝ}
    (hp : NonnegPrior p) (hp1 : NormalisedPrior p) {e : Θ → Z → ℝ}
    (he : StochasticKernel e) :
    0 ≤ caraInformedCE r x p e - caraActNowCE r x p := by
  have h0 : (fun a θ => x a θ - 0) = x := funext fun a => funext fun θ => sub_zero _
  have hevsi : caraActNowVal r x p ≤ caraInformedVal r x p e 0 := by
    unfold caraActNowVal caraInformedVal
    rw [h0]
    have h := evsi_nonneg (caraTable r x) p he.2
    unfold evsi at h
    linarith
  have hinf_pos : 0 < -caraInformedVal r x p e 0 :=
    neg_pos.mpr (caraInformedVal_zero_neg r x hp hp1 he)
  have hlog : Real.log (-caraInformedVal r x p e 0) ≤ Real.log (-caraActNowVal r x p) :=
    Real.log_le_log hinf_pos (by linarith)
  unfold caraInformedCE caraActNowCE
  have hinv : 0 < 1 / r := one_div_pos.mpr hr
  nlinarith

/-! ### Smooth ambiguity (Klibanoff, Marinacci & Mukerji 2005) -/

/-- General KMM smooth ambiguity score `φ⁻¹(∑ i, μ i · φ(E_{P i}[x]))`. -/
def smoothAmbiguity (φ φInv : ℝ → ℝ) (μ : I → ℝ) (P : I → Θ → ℝ) (x : Θ → ℝ) : ℝ :=
  φInv (∑ i, μ i * φ (∑ θ, P i θ * x θ))

/-- Exponential (constant absolute ambiguity aversion `η`) KMM smooth ambiguity score
`-(1 / η) * log (∑ i, μ i * exp(-η * E_{P i}[x]))`. -/
noncomputable def expSmoothAmbiguity (η : ℝ) (μ : I → ℝ) (P : I → Θ → ℝ) (x : Θ → ℝ) : ℝ :=
  caraCE η μ (fun i => ∑ θ, P i θ * x θ)

/-- Shift-invariant log-sum-exp identity backing `_aggregate(::SmoothAmbiguity, ...)`. -/
theorem expSmoothAmbiguity_logSumExp_eq {η : ℝ} (hη : η ≠ 0) {μ : I → ℝ}
    (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ) (P : I → Θ → ℝ) (x : Θ → ℝ) (m : ℝ) :
    expSmoothAmbiguity η μ P x =
      m - (1 / η) * Real.log (∑ i, μ i * Real.exp (-η * ((∑ θ, P i θ * x θ) - m))) :=
  caraCE_logSumExp_eq hη hμ hμ1 (fun i => ∑ θ, P i θ * x θ) m

/-- For `η > 0`, exponential smooth ambiguity is bounded below by the lower expectation over the
credal generators (`LowerExpectation`). -/
theorem lowerExpectedUtility_le_expSmoothAmbiguity {η : ℝ} (hη : 0 < η) {μ : I → ℝ}
    (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ) (u : A → Θ → ℝ) (P : I → Θ → ℝ) (a : A) :
    lowerExpectedUtility u P a ≤ expSmoothAmbiguity η μ P (u a) :=
  minOver_le_caraCE hη hμ hμ1 (fun i => ∑ θ, P i θ * u a θ)

/-- Ambiguity aversion: for `η > 0`, exponential smooth ambiguity is bounded above by the
expected utility under the second-order mixture prior `mixture μ P`. -/
theorem expSmoothAmbiguity_le_mixture {η : ℝ} (hη : 0 < η) {μ : I → ℝ}
    (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ) (P : I → Θ → ℝ) (x : Θ → ℝ) :
    expSmoothAmbiguity η μ P x ≤ ∑ θ, mixture μ P θ * x θ := by
  have hmix : ∑ θ, mixture μ P θ * x θ = ∑ i, μ i * ∑ θ, P i θ * x θ := by
    simp only [mixture, Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun θ _ => by ring
  rw [hmix]
  exact caraCE_le_mean hη hμ hμ1 (fun i => ∑ θ, P i θ * x θ)

/-- For `η > 0`, exponential smooth ambiguity is bounded above by the upper expectation over the
credal generators (`UpperExpectation`). -/
theorem expSmoothAmbiguity_le_upperExpectedUtility {η : ℝ} (hη : 0 < η) {μ : I → ℝ}
    (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ) (u : A → Θ → ℝ) (P : I → Θ → ℝ) (a : A) :
    expSmoothAmbiguity η μ P (u a) ≤ upperExpectedUtility u P a :=
  caraCE_le_maxOver hη hμ hμ1 (fun i => ∑ θ, P i θ * u a θ)

/-- Exponential smooth ambiguity is monotone in the payoff row for nonnegative generators. -/
theorem expSmoothAmbiguity_mono {η : ℝ} (hη : 0 < η) {μ : I → ℝ}
    (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ) {P : I → Θ → ℝ}
    (hP : ∀ i, NonnegPrior (P i)) {x y : Θ → ℝ} (hxy : ∀ θ, x θ ≤ y θ) :
    expSmoothAmbiguity η μ P x ≤ expSmoothAmbiguity η μ P y :=
  caraCE_mono hη hμ hμ1 fun i =>
    Finset.sum_le_sum fun θ _ => mul_le_mul_of_nonneg_left (hxy θ) (hP i θ)

/-- Exponential smooth ambiguity is translation-equivariant for normalised generators. -/
theorem expSmoothAmbiguity_add_const {η : ℝ} (hη : η ≠ 0) {μ : I → ℝ}
    (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ) {P : I → Θ → ℝ}
    (hP : ∀ i, NormalisedPrior (P i)) (x : Θ → ℝ) (c : ℝ) :
    expSmoothAmbiguity η μ P (fun θ => x θ + c) = expSmoothAmbiguity η μ P x + c := by
  have hrow : (fun i => ∑ θ, P i θ * (x θ + c)) = fun i => (∑ θ, P i θ * x θ) + c :=
    funext fun i => by
      simp only [mul_add, Finset.sum_add_distrib, ← Finset.sum_mul]
      rw [hP i, one_mul]
  unfold expSmoothAmbiguity
  rw [hrow]
  exact caraCE_add_const hη hμ hμ1 _ c

/-- Exponential smooth ambiguity preserves constants for normalised generators. -/
theorem expSmoothAmbiguity_const {η : ℝ} (hη : η ≠ 0) {μ : I → ℝ}
    (hμ1 : NormalisedPrior μ) {P : I → Θ → ℝ} (hP : ∀ i, NormalisedPrior (P i)) (c : ℝ) :
    expSmoothAmbiguity η μ P (fun _ => c) = c := by
  have hrow : (fun i => ∑ θ, P i θ * c) = fun _ => c :=
    funext fun i => by rw [← Finset.sum_mul, hP i, one_mul]
  unfold expSmoothAmbiguity
  rw [hrow]
  exact caraCE_const hη hμ1 c

/-- Exponential smooth ambiguity (`η > 0`) is concave in the gamble `x` (preference for hedging
against ambiguity). -/
theorem expSmoothAmbiguity_concave {η : ℝ} (hη : 0 < η) {μ : I → ℝ}
    (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ) (P : I → Θ → ℝ) (x y : Θ → ℝ)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    (1 - t) * expSmoothAmbiguity η μ P x + t * expSmoothAmbiguity η μ P y ≤
      expSmoothAmbiguity η μ P (fun θ => (1 - t) * x θ + t * y θ) := by
  have hrow : (fun i => ∑ θ, P i θ * ((1 - t) * x θ + t * y θ)) =
      fun i => (1 - t) * (∑ θ, P i θ * x θ) + t * (∑ θ, P i θ * y θ) :=
    funext fun i => by
      have h1 : ∀ θ, P i θ * ((1 - t) * x θ + t * y θ) =
          (1 - t) * (P i θ * x θ) + t * (P i θ * y θ) := fun θ => by ring
      rw [Finset.sum_congr rfl fun θ _ => h1 θ, Finset.sum_add_distrib,
        ← Finset.mul_sum, ← Finset.mul_sum]
  unfold expSmoothAmbiguity
  rw [hrow]
  exact caraCE_concave hη hμ hμ1 _ _ ht0 ht1

/-- General KMM ambiguity aversion: for any strictly increasing concave `φ` with left inverse
`φInv`, `smoothAmbiguity φ φInv μ P x` never exceeds the `μ`-mixture expectation. -/
theorem smoothAmbiguity_le_mixture_of_concave {φ φInv : ℝ → ℝ}
    (hconc : ConcaveOn ℝ Set.univ φ) (hmono : StrictMono φ)
    (hinv : ∀ y, φInv (φ y) = y) (hinvMono : Monotone φInv)
    {μ : I → ℝ} (hμ : NonnegPrior μ) (hμ1 : NormalisedPrior μ)
    (P : I → Θ → ℝ) (x : Θ → ℝ) :
    smoothAmbiguity φ φInv μ P x ≤ ∑ θ, mixture μ P θ * x θ := by
  have hjensen : ∑ i, μ i * φ (∑ θ, P i θ * x θ) ≤ φ (∑ i, μ i * ∑ θ, P i θ * x θ) :=
    hconc.le_map_sum (fun i _ => hμ i) hμ1 (fun _ _ => Set.mem_univ _)
  have hmix : ∑ θ, mixture μ P θ * x θ = ∑ i, μ i * ∑ θ, P i θ * x θ := by
    simp only [mixture, Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun θ _ => by ring
  unfold smoothAmbiguity
  have _ := hmono
  calc φInv (∑ i, μ i * φ (∑ θ, P i θ * x θ))
      ≤ φInv (φ (∑ i, μ i * ∑ θ, P i θ * x θ)) := hinvMono hjensen
    _ = ∑ i, μ i * ∑ θ, P i θ * x θ := hinv _
    _ = ∑ θ, mixture μ P θ * x θ := hmix.symm

end CARA

end DecisionAnalysisProofs
