import DecisionAnalysisProofs.Finite.Credal
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!
# DecisionAnalysisProofs.Finite.Envelope

The envelope of the single-prior EVPI over a credal set (T15), as computed by `robust_evpi`
(`src/voi.jl`): the fields `lower` and `upper` of `RobustVOI`, and the programme
`evpi_envelope_model` (`src/programs.jl`).

The credal set is the convex hull of a finite nonempty family `P : I → Θ → K`, as in
`Finite/Credal.lean`; a member of the hull is `mixture w P` with `w` nonnegative and summing to
one. `members[k]` in Julia is `evpi u (P k)`.

## What each result backs

* **`sum_mul_evpi_le_evpi_mixture`**: EVPI is concave in the prior (the perfect-information
  term is linear, `perfectValue_mixture`; the act-now term is convex,
  `actNowValue_mixture_le`). Backs the sentence "EVPI is concave in the prior" in the
  docstrings of `robust_evpi` and `evpi_envelope_model`. Read with `w` the sampling
  distribution of an unbiased estimate of the belief, it also says that EVPI computed from a
  sampled belief is biased downwards, the act-now term being overstated
  (`actNowValue_mixture_le`): the selection-bias note of `DecisionReport`.
* **`minOver_evpi_le_evpi_mixture`**, **`exists_forall_evpi_le_evpi_mixture`**: the smallest
  EVPI over the hull is attained at a generating point. Backs
  `lower = minimum(members)` in `robust_evpi` and the assertions
  `v.lower == minimum(members)` and `v.lower <= minimum(grid)` of `test/test_voi.jl`.
* The programme of `evpi_envelope_model`, maximise `Σ_k λ_k m_k − w` subject to `λ` in the
  simplex and `Σ_k λ_k E[a, k] ≤ w` for every action, is `EvpiEnvelopeFeasible` with objective
  `evpiEnvelopeObjective`. **`EvpiEnvelopeFeasible.objective_le`**: every feasible point's
  objective is at most the EVPI at its own mixture. **`evpiEnvelopeFeasible_actNowValue`**:
  every hull prior gives a feasible point whose objective is its EVPI.
  **`evpiEnvelope_optimum`**: therefore an optimal solution's objective *is* the EVPI at the
  prior `Σ_k λ_k p_k`, and is the largest EVPI over the hull. Backs
  `upper = _evpi_upper(...)`, the docstring "at the optimum `w` is the act-now value under the
  prior `Σ_k λ_k p_k`, so the objective is the EVPI at that prior", and the assertions
  `v.upper >= maximum(members)` and `v.upper >= maximum(grid)`.
* **`evpi_max_not_at_generating_point`**: an instance in which EVPI is zero at both
  generating points and `1/2` at their midpoint. Backs the design decision that the upper end
  needs the programme and cannot be `maximum(members)`.

Not claimed: that the programme has an optimal solution (it does, being feasible and bounded,
but linear-programming attainment is not proved here); anything about EVSI, whose envelope
Julia does not compute; anything about the rule-specific `value` of `RobustVOI`.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ I : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype I] [Nonempty I]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

omit [Nonempty I] in
/-- The perfect-information value is linear in the prior. -/
theorem perfectValue_mixture (u : A → Θ → K) (w : I → K) (P : I → Θ → K) :
    perfectValue u (mixture w P) = ∑ i, w i * perfectValue u (P i) := by
  simp only [perfectValue, mixture, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun θ _ => by ring

omit [Nonempty I] in
/-- The act-now value is convex in the prior. -/
theorem actNowValue_mixture_le (u : A → Θ → K) {w : I → K} (hw : ∀ i, 0 ≤ w i)
    (P : I → Θ → K) : actNowValue u (mixture w P) ≤ ∑ i, w i * actNowValue u (P i) := by
  unfold actNowValue
  rw [show expectedUtility u (mixture w P) = fun a => ∑ i, w i * expectedUtility u (P i) a from
    funext (expectedUtility_mixture u w P)]
  exact maxOver_weighted_sum_le w hw _

omit [Nonempty I] in
/-- **T15.** EVPI is concave in the prior. -/
theorem sum_mul_evpi_le_evpi_mixture (u : A → Θ → K) {w : I → K} (hw : ∀ i, 0 ≤ w i)
    (P : I → Θ → K) : ∑ i, w i * evpi u (P i) ≤ evpi u (mixture w P) := by
  have h := actNowValue_mixture_le u hw P
  simp only [evpi, mul_sub, Finset.sum_sub_distrib, perfectValue_mixture]
  linarith

/-- **T15** (`lower = minimum(members)`). The smallest EVPI among the generating points is a
lower bound for the EVPI at every prior of the hull. -/
theorem minOver_evpi_le_evpi_mixture (u : A → Θ → K) {w : I → K} (hw : ∀ i, 0 ≤ w i)
    (hw1 : ∑ i, w i = 1) (P : I → Θ → K) :
    minOver (fun i => evpi u (P i)) ≤ evpi u (mixture w P) := by
  calc minOver (fun i => evpi u (P i))
      = ∑ i, w i * minOver (fun i => evpi u (P i)) := by rw [← Finset.sum_mul, hw1, one_mul]
    _ ≤ ∑ i, w i * evpi u (P i) :=
        Finset.sum_le_sum fun i _ =>
          mul_le_mul_of_nonneg_left (minOver_le (fun i => evpi u (P i)) i) (hw i)
    _ ≤ evpi u (mixture w P) := sum_mul_evpi_le_evpi_mixture u hw P

/-- **T15.** The minimum of EVPI over the hull is attained at a generating point. -/
theorem exists_forall_evpi_le_evpi_mixture (u : A → Θ → K) (P : I → Θ → K) :
    ∃ i, ∀ w : I → K, (∀ j, 0 ≤ w j) → ∑ j, w j = 1 →
      evpi u (P i) ≤ evpi u (mixture w P) := by
  obtain ⟨i, hi⟩ := exists_eq_minOver (fun i => evpi u (P i))
  exact ⟨i, fun w hw hw1 => hi ▸ minOver_evpi_le_evpi_mixture u hw hw1 P⟩

/-! ### The programme of `evpi_envelope_model` -/

/-- A feasible point `(λ, w)` of the programme of `evpi_envelope_model`: `λ` in the simplex
and `Σ_k λ_k E[a, k] ≤ w` for every action. -/
def EvpiEnvelopeFeasible (u : A → Θ → K) (P : I → Θ → K) (lam : I → K) (t : K) : Prop :=
  (∀ k, 0 ≤ lam k) ∧ ∑ k, lam k = 1 ∧ ∀ a, ∑ k, lam k * expectedUtility u (P k) a ≤ t

/-- The objective `Σ_k λ_k m_k − w` of the programme, with `m_k` the perfect-information value
at the generating point `k`. -/
def evpiEnvelopeObjective (u : A → Θ → K) (P : I → Θ → K) (lam : I → K) (t : K) : K :=
  ∑ k, lam k * perfectValue u (P k) - t

omit [Nonempty I] in
/-- The objective of a feasible point is at most the EVPI at its mixture. -/
theorem EvpiEnvelopeFeasible.objective_le {u : A → Θ → K} {P : I → Θ → K} {lam : I → K}
    {t : K} (h : EvpiEnvelopeFeasible u P lam t) :
    evpiEnvelopeObjective u P lam t ≤ evpi u (mixture lam P) := by
  have h1 : actNowValue u (mixture lam P) ≤ t :=
    maxOver_le fun a => by rw [expectedUtility_mixture]; exact h.2.2 a
  unfold evpiEnvelopeObjective evpi
  rw [perfectValue_mixture]
  linarith

omit [Nonempty I] in
/-- Every prior of the hull, with its act-now value, is a feasible point whose objective is
its EVPI. -/
theorem evpiEnvelopeFeasible_actNowValue (u : A → Θ → K) (P : I → Θ → K) {lam : I → K}
    (hl : ∀ k, 0 ≤ lam k) (hl1 : ∑ k, lam k = 1) :
    EvpiEnvelopeFeasible u P lam (actNowValue u (mixture lam P)) ∧
      evpiEnvelopeObjective u P lam (actNowValue u (mixture lam P)) =
        evpi u (mixture lam P) := by
  refine ⟨⟨hl, hl1, fun a => ?_⟩, ?_⟩
  · rw [← expectedUtility_mixture]
    exact expectedUtility_le_actNowValue u _ a
  · unfold evpiEnvelopeObjective evpi
    rw [perfectValue_mixture]

omit [Nonempty I] in
/-- **T15** (`upper`). An optimal solution of the programme has objective equal to the EVPI at
the prior `Σ_k λ_k p_k`, and that is the largest EVPI over the hull; in particular it is at
least the EVPI at every generating point. -/
theorem evpiEnvelope_optimum {u : A → Θ → K} {P : I → Θ → K} {lam : I → K} {t : K}
    (hf : EvpiEnvelopeFeasible u P lam t)
    (hopt : ∀ lam' t', EvpiEnvelopeFeasible u P lam' t' →
      evpiEnvelopeObjective u P lam' t' ≤ evpiEnvelopeObjective u P lam t) :
    evpiEnvelopeObjective u P lam t = evpi u (mixture lam P) ∧
      (∀ w : I → K, (∀ k, 0 ≤ w k) → ∑ k, w k = 1 →
        evpi u (mixture w P) ≤ evpiEnvelopeObjective u P lam t) ∧
      ∀ i, evpi u (P i) ≤ evpiEnvelopeObjective u P lam t := by
  classical
  have hull : ∀ w : I → K, (∀ k, 0 ≤ w k) → ∑ k, w k = 1 →
      evpi u (mixture w P) ≤ evpiEnvelopeObjective u P lam t := fun w hw hw1 => by
    obtain ⟨hfeas, hobj⟩ := evpiEnvelopeFeasible_actNowValue u P hw hw1
    rw [← hobj]
    exact hopt _ _ hfeas
  refine ⟨le_antisymm hf.objective_le (hull lam hf.1 hf.2.1), hull, fun i => ?_⟩
  have hm : mixture (fun j => if j = i then (1 : K) else 0) P = P i := by
    funext θ
    simp [mixture]
  have h := hull (fun j => if j = i then 1 else 0)
    (fun j => by by_cases hj : j = i <;> simp [hj]) (by simp)
  rwa [hm] at h

/-! ### The maximum is not at a generating point -/

/-- Guess the state: payoff one for a correct guess. -/
def guessU : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 1 | 0, 1 => 0
  | 1, 0 => 0 | 1, 1 => 1

/-- The two point masses. -/
def pointMasses : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 1 | 0, 1 => 0
  | 1, 0 => 0 | 1, 1 => 1

/-- EVPI is zero at both generating points and `1/2` at their midpoint: the upper end of the
envelope is not `maximum(members)`. -/
theorem evpi_max_not_at_generating_point :
    (∀ i, evpi guessU (pointMasses i) = 0) ∧
      evpi guessU (mixture (fun _ => 1/2) pointMasses) = 1/2 := by
  constructor
  · intro i
    fin_cases i <;>
      norm_num [evpi, perfectValue, actNowValue, expectedUtility, maxOver_fin_two,
        Fin.sum_univ_two, guessU, pointMasses]
  · norm_num [evpi, perfectValue, actNowValue, expectedUtility, mixture, maxOver_fin_two,
      Fin.sum_univ_two, guessU, pointMasses]

end DecisionAnalysisProofs
