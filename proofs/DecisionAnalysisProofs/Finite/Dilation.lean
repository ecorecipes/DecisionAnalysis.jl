import DecisionAnalysisProofs.Finite.Credal
import DecisionAnalysisProofs.Finite.Garbling
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!
# DecisionAnalysisProofs.Finite.Dilation

Free information can have negative value under Γ-maximin (T13). This is the instance of the
Julia test "robust EVSI can be negative and is not clamped" (`test/test_voi.jl`), with the
quantities `robust_evsi` (`src/voi.jl`) computes for it.

## What `robust_evsi` computes, in Lean

For a family of priors `P`, an experiment `e` and the rule Γ-maximin:

* `act_now` is `gammaMaximinValue u P` (`Finite/Credal.lean`).
* After signal `z` every prior is updated and the rule is applied to the updated set:
  `updatedLowerUtility u P e z a` is the smallest posterior expected utility of `a`,
  `min_i signalValue u (P i) e z a / signalProbability (P i) e z`. Julia drops the priors that
  give the signal probability zero; this definition agrees with it when every prior gives
  every signal positive probability, which is part of the theorem below for the instance.
* The choices form a strategy `d : Z → A`; `strategyPayoff u e d θ = Σ_z e θ z · u (d z) θ` is
  Julia's vector `w`, and `informed` is `strategyLowerValue u P e d`, its lower expectation
  over the *prior* set.

## Result

* **T13**, **`dilation_robust_evsi_negative`**: for the dilation instance (a fair coin `X`, an
  unknown `Y`, the signal `X xor Y`), `act_now = 1/10`; every prior gives every signal
  positive probability; after either signal "abstain" has updated lower utility `0` and both
  bets `-1`, so the strategy is "always abstain"; `informed = 0`; and yet each prior taken
  alone gives the signal EVSI `11/10 > 0`. So `value = informed − act_now = −1/10 < 0`.

Backs the test's figures (`v.act_now == 1//10`, `v.informed == 0`, `v.value == -1//10`,
`all(>(0), v.members)`) and the documented behaviour that `robust_evsi` returns a negative
value unclamped. Shapes the package: Blackwell monotonicity (T6) and `evsi_nonneg` must not be
assumed for a credal set, and a report must not clamp or hide a negative robust value.

Not claimed: anything about rules other than Γ-maximin, about tie-breaking by `first_choice`
(the maximiser is unique here), or the commitment variant (T13b in `Roadmap.lean`).
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ I Z : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype I] [Nonempty I]
  [Fintype Z] {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The smallest posterior expected utility of `a` after signal `z`, over the updated priors
(the score `robust_evsi` maximises after each signal, under `GammaMaximin`). -/
def updatedLowerUtility (u : A → Θ → K) (P : I → Θ → K) (e : Θ → Z → K) (z : Z) (a : A) : K :=
  minOver fun i => signalValue u (P i) e z a / signalProbability (P i) e z

/-- The state-contingent expected payoff of the strategy `d` (the vector `w` of
`robust_evsi`). -/
def strategyPayoff (u : A → Θ → K) (e : Θ → Z → K) (d : Z → A) : Θ → K :=
  fun θ => ∑ z, e θ z * u (d z) θ

/-- The lower expectation over the prior set of the strategy `d` (`informed` in
`robust_evsi`). -/
def strategyLowerValue (u : A → Θ → K) (P : I → Θ → K) (e : Θ → Z → K) (d : Z → A) : K :=
  minOver fun i => ∑ θ, P i θ * strategyPayoff u e d θ

/-! ### The dilation instance -/

/-- Rows: bet on one, bet on zero, abstain. Columns: `x0y0`, `x1y0`, `x0y1`, `x1y1`. -/
def dilationU : Fin 3 → Fin 4 → ℚ
  | 0, 0 => -1 | 0, 1 => 6/5 | 0, 2 => -1 | 0, 3 => 6/5
  | 1, 0 => 6/5 | 1, 1 => -1 | 1, 2 => 6/5 | 1, 3 => -1
  | 2, 0 => 0 | 2, 1 => 0 | 2, 2 => 0 | 2, 3 => 0

/-- The two generating priors: `X` is a fair coin, `Y` is 0 or 1. -/
def dilationP : Fin 2 → Fin 4 → ℚ
  | 0, 0 => 1/2 | 0, 1 => 1/2 | 0, 2 => 0 | 0, 3 => 0
  | 1, 0 => 0 | 1, 1 => 0 | 1, 2 => 1/2 | 1, 3 => 1/2

/-- The signal `X xor Y`: signal 0 is "same", signal 1 is "different". -/
def dilationE : Fin 4 → Fin 2 → ℚ
  | 0, 0 => 1 | 0, 1 => 0
  | 1, 0 => 0 | 1, 1 => 1
  | 2, 0 => 0 | 2, 1 => 1
  | 3, 0 => 1 | 3, 1 => 0

/-- **T13.** Under Γ-maximin the signal has value `0 − 1/10 < 0`, although each prior alone
values it at `11/10`. -/
theorem dilation_robust_evsi_negative :
    gammaMaximinValue dilationU dilationP = 1/10 ∧
    (∀ i z, 0 < signalProbability (dilationP i) dilationE z) ∧
    (∀ z, updatedLowerUtility dilationU dilationP dilationE z 0 = -1 ∧
      updatedLowerUtility dilationU dilationP dilationE z 1 = -1 ∧
      updatedLowerUtility dilationU dilationP dilationE z 2 = 0) ∧
    strategyLowerValue dilationU dilationP dilationE (fun _ => 2) = 0 ∧
    (∀ i, evsi dilationU (dilationP i) dilationE = 11/10) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · norm_num [gammaMaximinValue, lowerExpectedUtility, expectedUtility, maxOver_fin_three,
      minOver_fin_two, Fin.sum_univ_four, dilationU, dilationP]
  · intro i z
    fin_cases i <;> fin_cases z <;>
      norm_num [signalProbability, Fin.sum_univ_four, dilationP, dilationE]
  · intro z
    fin_cases z <;>
      norm_num [updatedLowerUtility, signalValue, signalProbability, minOver_fin_two,
        Fin.sum_univ_four, dilationU, dilationP, dilationE]
  · norm_num [strategyLowerValue, strategyPayoff, minOver_fin_two, Fin.sum_univ_four,
      Fin.sum_univ_two, dilationU, dilationP, dilationE]
  · intro i
    fin_cases i <;>
      norm_num [evsi, valueWithInformation, actNowValue, signalValue, expectedUtility,
        maxOver_fin_three, Fin.sum_univ_four, Fin.sum_univ_two, dilationU, dilationP,
        dilationE]

end DecisionAnalysisProofs
