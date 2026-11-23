import DecisionAnalysisProofs.Finite.Problem
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!
# DecisionAnalysisProofs.Finite.SelectionBias

What maximising over **estimated** payoffs does to the reported values (T16, the optimiser's
curse). The payoff table handed to `DecisionAnalysis.jl` is often an estimate (a mean over a
few simulator replicates per cell). This file models the estimate as a finite distribution
over tables: scenarios `ω : Ω` with weights `w ω`, estimated tables `U ω : A → Θ → K`, and the
table of cell means `meanTable w U`. When the estimate is unbiased cell by cell,
`meanTable w U` is the true table.

## What each result backs or shapes

* **T16**, **`actNowValue_meanTable_le`**: the expected reported act-now value
  (`act_now_value`, the best score of `decide(problem, MaxExpectedUtility())`) is at least the
  true one. **`perfectValue_meanTable_le`** and **`valueWithInformation_meanTable_le`**: the
  same for the informed values inside `evpi` and `value_with_information`. Each needs only
  nonnegative weights.
* **`sum_mul_expectedUtility_meanTable_le`**: the action chosen from the estimated table is,
  on average, worth less under the true table than its reported score (post-decision
  disappointment). This is the statement a report should make about the chosen action.
* **`evpi_bias_sign_indefinite`**: the bias of EVPI itself has **no definite sign**. `evpi` is
  a difference of two upward-biased terms; one instance has expected estimated EVPI above the
  true EVPI and another has it below. This contradicts the sentence in the docstring of
  `evpi` (`src/voi.jl`), "the maximum over noisy cells biases EVPI upwards", which is true of
  the perfect-information term alone. Shapes the package: a selection-bias note must be
  attached to the *values* (act-now, informed) and to the chosen action, not phrased as an
  upward bias of EVPI or EVSI.

The other source of selection bias, a **belief** estimated from a sample with the payoffs
exact, is covered by `Finite/Envelope.lean`: the act-now value is convex in the prior
(`actNowValue_mixture_le`) and the perfect-information value is linear
(`perfectValue_mixture`), so there EVPI is biased **downwards**
(`sum_mul_evpi_le_evpi_mixture`). That backs the report note "maximising sample means
overstates the act-now value" (`src/report.jl`); for EVSI and EVPPI the informed term is
convex in the prior as well, so their bias under a sampled belief again has no proved sign.

The inequality is `maxOver_weighted_sum_le` (a mixture of maxima dominates the maximum of the
mixture), the same one that gives Blackwell's easy direction.

Not claimed: any size for the bias, anything about a particular estimator or about
independence between cells, or any correction.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {Ω A Θ Z : Type} [Fintype Ω] [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The table of cell means of the estimated tables `U ω` under the weights `w`. -/
def meanTable (w : Ω → K) (U : Ω → A → Θ → K) : A → Θ → K := fun a θ => ∑ ω, w ω * U ω a θ

omit [Nonempty A] in
/-- Expected utility is linear in the payoff table. -/
theorem expectedUtility_meanTable (w : Ω → K) (U : Ω → A → Θ → K) (p : Θ → K) (a : A) :
    expectedUtility (meanTable w U) p a = ∑ ω, w ω * expectedUtility (U ω) p a := by
  simp only [expectedUtility, meanTable, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun ω _ => Finset.sum_congr rfl fun θ _ => by ring

omit [Nonempty A] in
theorem signalValue_meanTable (w : Ω → K) (U : Ω → A → Θ → K) (p : Θ → K) (e : Θ → Z → K)
    (z : Z) (a : A) :
    signalValue (meanTable w U) p e z a = ∑ ω, w ω * signalValue (U ω) p e z a := by
  simp only [signalValue, meanTable, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun ω _ => Finset.sum_congr rfl fun θ _ => by ring

/-- **T16.** The act-now value of the true (mean) table is at most the expected act-now value
of the estimated tables: the reported best score is biased upwards. -/
theorem actNowValue_meanTable_le {w : Ω → K} (hw : ∀ ω, 0 ≤ w ω) (U : Ω → A → Θ → K)
    (p : Θ → K) : actNowValue (meanTable w U) p ≤ ∑ ω, w ω * actNowValue (U ω) p := by
  unfold actNowValue
  rw [show expectedUtility (meanTable w U) p =
    fun a => ∑ ω, w ω * expectedUtility (U ω) p a from
      funext (expectedUtility_meanTable w U p)]
  exact maxOver_weighted_sum_le w hw _

/-- **T16**, for the perfect-information value inside `evpi`. -/
theorem perfectValue_meanTable_le {w : Ω → K} (hw : ∀ ω, 0 ≤ w ω) (U : Ω → A → Θ → K)
    {p : Θ → K} (hp : NonnegPrior p) :
    perfectValue (meanTable w U) p ≤ ∑ ω, w ω * perfectValue (U ω) p := by
  unfold perfectValue
  calc ∑ θ, p θ * maxOver (fun a => meanTable w U a θ)
      ≤ ∑ θ, p θ * ∑ ω, w ω * maxOver (fun a => U ω a θ) :=
        Finset.sum_le_sum fun θ _ => mul_le_mul_of_nonneg_left
          (maxOver_weighted_sum_le w hw fun ω a => U ω a θ) (hp θ)
    _ = ∑ ω, w ω * ∑ θ, p θ * maxOver (fun a => U ω a θ) := by
        simp only [Finset.mul_sum]
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun ω _ => Finset.sum_congr rfl fun θ _ => by ring

/-- **T16**, for `value_with_information`. -/
theorem valueWithInformation_meanTable_le {w : Ω → K} (hw : ∀ ω, 0 ≤ w ω)
    (U : Ω → A → Θ → K) (p : Θ → K) (e : Θ → Z → K) :
    valueWithInformation (meanTable w U) p e ≤ ∑ ω, w ω * valueWithInformation (U ω) p e := by
  unfold valueWithInformation
  calc ∑ z, maxOver (signalValue (meanTable w U) p e z)
      ≤ ∑ z, ∑ ω, w ω * maxOver (signalValue (U ω) p e z) :=
        Finset.sum_le_sum fun z _ => by
          rw [show signalValue (meanTable w U) p e z =
            fun a => ∑ ω, w ω * signalValue (U ω) p e z a from
              funext (signalValue_meanTable w U p e z)]
          exact maxOver_weighted_sum_le w hw _
    _ = ∑ ω, w ω * ∑ z, maxOver (signalValue (U ω) p e z) := by
        simp only [Finset.mul_sum]
        exact Finset.sum_comm

/-- **Optimiser's curse.** If in every scenario the chosen action `c ω` is Bayes for the
estimated table, then on average its true expected utility is at most its reported one. -/
theorem sum_mul_expectedUtility_meanTable_le {w : Ω → K} (hw : ∀ ω, 0 ≤ w ω)
    (hw1 : ∑ ω, w ω = 1) (U : Ω → A → Θ → K) (p : Θ → K) {c : Ω → A}
    (hc : ∀ ω, IsBayes (U ω) p (c ω)) :
    ∑ ω, w ω * expectedUtility (meanTable w U) p (c ω) ≤
      ∑ ω, w ω * expectedUtility (U ω) p (c ω) := by
  calc ∑ ω, w ω * expectedUtility (meanTable w U) p (c ω)
      ≤ ∑ ω, w ω * actNowValue (meanTable w U) p :=
        Finset.sum_le_sum fun ω _ => mul_le_mul_of_nonneg_left
          (expectedUtility_le_actNowValue _ p (c ω)) (hw ω)
    _ = actNowValue (meanTable w U) p := by rw [← Finset.sum_mul, hw1, one_mul]
    _ ≤ ∑ ω, w ω * actNowValue (U ω) p := actNowValue_meanTable_le hw U p
    _ = ∑ ω, w ω * expectedUtility (U ω) p (c ω) :=
        Finset.sum_congr rfl fun ω _ => by rw [(isBayes_iff (U ω) p (c ω)).mp (hc ω)]

/-! ### The bias of EVPI has no sign -/

/-- Two equally likely scenarios. -/
def halves : Fin 2 → ℚ := fun _ => 1/2

/-- Estimated tables whose cell means are all zero: the true EVPI is zero, each estimate
reports one. -/
def biasUpU : Fin 2 → Fin 2 → Fin 2 → ℚ
  | 0, 0, 0 => 1 | 0, 0, 1 => -1 | 0, 1, 0 => -1 | 0, 1, 1 => 1
  | 1, 0, 0 => -1 | 1, 0, 1 => 1 | 1, 1, 0 => 1 | 1, 1, 1 => -1

/-- Estimated tables in which only the cell (action 0, state 0) is noisy, `10 ± 2`. The true
table has the two actions tied under the uniform prior. -/
def biasDownU : Fin 2 → Fin 2 → Fin 2 → ℚ
  | 0, 0, 0 => 12 | 0, 0, 1 => 0 | 0, 1, 0 => 0 | 0, 1, 1 => 10
  | 1, 0, 0 => 8 | 1, 0, 1 => 0 | 1, 1, 0 => 0 | 1, 1, 1 => 10

/-- The expected EVPI of unbiased estimated tables can be above the true EVPI (`1 > 0`) and
can be below it (`9/2 < 5`): the docstring claim that noise "biases EVPI upwards" does not
hold in general. -/
theorem evpi_bias_sign_indefinite :
    (evpi (meanTable halves biasUpU) halves = 0 ∧
      ∑ ω, halves ω * evpi (biasUpU ω) halves = 1) ∧
    (evpi (meanTable halves biasDownU) halves = 5 ∧
      ∑ ω, halves ω * evpi (biasDownU ω) halves = 9/2) := by
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;>
    norm_num [evpi, perfectValue, actNowValue, expectedUtility, meanTable, maxOver_fin_two,
      Fin.sum_univ_two, halves, biasUpU, biasDownU]

end DecisionAnalysisProofs
