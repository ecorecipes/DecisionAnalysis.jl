import DecisionAnalysisProofs.Finite.Garbling
import DecisionAnalysisProofs.Finite.Credal
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!
# DecisionAnalysisProofs.Finite.Examples

**T9**: the three worked examples that are the shared fixtures of `DecisionAnalysis.jl`
(`test/fixtures/`, checked against `lab/wiki/references/computations/worked_examples.py`),
over `ℚ`. Every figure quoted for them is a theorem here, about the definitions of
`Finite/Problem.lean`, `Finite/Garbling.lean` and `Finite/Credal.lean` instantiated at
`K = ℚ`; nothing is evaluated by `#eval` or `native_decide`. Each proof unfolds the
definitions with `norm_num` (finite sums over `Fin n`, `maxOver`/`minOver` as nested
`max`/`min`).

## Flood defence (3 actions × 4 states, 3-member ensemble, river gauge)

Actions: 0 do nothing, 1 raise levee, 2 relocate. Signals: 0 gauge low, 1 gauge high.

| Quantity | Value | Theorem |
|:--|:--|:--|
| expected utility | −39/2, −97/4, −48 | `flood_expectedUtility` |
| Bayes action | 0 only | `flood_isBayes` |
| worst case (maximin) | −150, −90, −48 | `flood_worstCase` |
| max regret | 102, 42, 48; minimax regret picks 1 | `flood_maxRegret`, `flood_isMinimaxRegret` |
| expected regret | 207/20, 151/10, 777/20 | `flood_expectedRegret` |
| lower expected utility | −33, −113/4, −48; Γ-maximin picks 1 | `flood_lowerExpectedUtility`, `flood_gammaMaximinValue`, `flood_isGammaMaximin` |
| E-admissible set | {0, 1} | `flood_isEAdmissible` |
| EVPI | 207/20 | `flood_evpi` |
| EVSI of the gauge | 97/20 | `flood_evsi_gauge` |
| EVSI of the garbled gauge | 49/25 | `flood_evsi_garbled` |

## Two-state epidemic (80/80 test)

EVPI 18, EVSI 44/5, net of the test cost 3: 29/5 (`epidemic_evpi`, `epidemic_evsi`,
`epidemic_net`).

## Equipment hazard (80/90 test)

EVPI 140, EVSI 78, net of the test cost 30: 48, expected regret (240, 140), max regret
(800, 200) (`equipment_evpi`, `equipment_evsi`, `equipment_net`, `equipment_expectedRegret`,
`equipment_maxRegret`).

Not claimed: that Julia's `Float64` results equal these rationals; the Julia tests compare
within a tolerance, and the rational fixtures are compared exactly by `check_certificate`.
-/

set_option autoImplicit false

namespace DecisionAnalysisProofs.Examples

open DecisionAnalysisProofs

/-! ### Flood defence -/

/-- Payoff table: rows do nothing, raise levee, relocate; columns the four flood states. -/
def floodU : Fin 3 → Fin 4 → ℚ
  | 0, 0 => 0 | 0, 1 => -10 | 0, 2 => -60 | 0, 3 => -150
  | 1, 0 => -20 | 1, 1 => -20 | 1, 2 => -25 | 1, 3 => -90
  | 2, 0 => -48 | 2, 1 => -48 | 2, 2 => -48 | 2, 3 => -48

/-- The central prior (0.50, 0.30, 0.15, 0.05). -/
def floodPrior : Fin 4 → ℚ
  | 0 => 1/2 | 1 => 3/10 | 2 => 3/20 | 3 => 1/20

/-- The three-member ensemble of priors; member 0 is the central prior. -/
def floodEnsemble : Fin 3 → Fin 4 → ℚ
  | 0, 0 => 1/2 | 0, 1 => 3/10 | 0, 2 => 3/20 | 0, 3 => 1/20
  | 1, 0 => 7/20 | 1, 1 => 3/10 | 1, 2 => 1/4 | 1, 3 => 1/10
  | 2, 0 => 3/5 | 2, 1 => 1/4 | 2, 2 => 3/25 | 2, 3 => 3/100

/-- The river gauge: `gauge θ z = P(z | θ)`, signal 0 low and 1 high. -/
def gauge : Fin 4 → Fin 2 → ℚ
  | 0, 0 => 9/10 | 0, 1 => 1/10
  | 1, 0 => 3/5 | 1, 1 => 2/5
  | 2, 0 => 1/5 | 2, 1 => 4/5
  | 3, 0 => 1/20 | 3, 1 => 19/20

/-- The garbling: the gauge reading is flipped with probability 1/5. -/
def flipKernel : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 4/5 | 0, 1 => 1/5
  | 1, 0 => 1/5 | 1, 1 => 4/5

theorem floodPrior_nonneg : NonnegPrior floodPrior := by
  intro θ
  fin_cases θ <;> norm_num [floodPrior]

theorem floodPrior_normalised : NormalisedPrior floodPrior := by
  norm_num [NormalisedPrior, Fin.sum_univ_four, floodPrior]

theorem floodEnsemble_zero : floodEnsemble 0 = floodPrior := by
  funext θ
  fin_cases θ <;> rfl

theorem gauge_stochastic : StochasticKernel gauge := by
  constructor
  · intro θ z
    fin_cases θ <;> fin_cases z <;> norm_num [gauge]
  · intro θ
    fin_cases θ <;> norm_num [Fin.sum_univ_two, gauge]

theorem flipKernel_stochastic : StochasticKernel flipKernel := by
  constructor
  · intro z z'
    fin_cases z <;> fin_cases z' <;> norm_num [flipKernel]
  · intro z
    fin_cases z <;> norm_num [Fin.sum_univ_two, flipKernel]

theorem flood_expectedUtility :
    expectedUtility floodU floodPrior 0 = -39/2 ∧
    expectedUtility floodU floodPrior 1 = -97/4 ∧
    expectedUtility floodU floodPrior 2 = -48 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [expectedUtility, Fin.sum_univ_four, floodU, floodPrior]

theorem flood_actNowValue : actNowValue floodU floodPrior = -39/2 := by
  obtain ⟨h0, h1, h2⟩ := flood_expectedUtility
  norm_num [actNowValue, maxOver_fin_three, h0, h1, h2]

/-- "Do nothing" is the unique Bayes action. -/
theorem flood_isBayes :
    IsBayes floodU floodPrior 0 ∧ ¬ IsBayes floodU floodPrior 1 ∧
      ¬ IsBayes floodU floodPrior 2 := by
  obtain ⟨h0, h1, h2⟩ := flood_expectedUtility
  simp only [isBayes_iff, flood_actNowValue, h0, h1, h2]
  norm_num

theorem flood_worstCase :
    worstCase floodU 0 = -150 ∧ worstCase floodU 1 = -90 ∧ worstCase floodU 2 = -48 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [worstCase, minOver_fin_four, floodU]

theorem flood_maxRegret :
    maxRegret floodU 0 = 102 ∧ maxRegret floodU 1 = 42 ∧ maxRegret floodU 2 = 48 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [maxRegret, regret, maxOver_fin_three, maxOver_fin_four, floodU]

/-- "Raise levee" is the unique minimax-regret action. -/
theorem flood_isMinimaxRegret :
    ¬ IsMinimaxRegret floodU 0 ∧ IsMinimaxRegret floodU 1 ∧ ¬ IsMinimaxRegret floodU 2 := by
  obtain ⟨h0, h1, h2⟩ := flood_maxRegret
  unfold IsMinimaxRegret
  norm_num [forall_fin_three, h0, h1, h2]

theorem flood_expectedRegret :
    expectedRegret floodU floodPrior 0 = 207/20 ∧
    expectedRegret floodU floodPrior 1 = 151/10 ∧
    expectedRegret floodU floodPrior 2 = 777/20 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [expectedRegret, regret, maxOver_fin_three, Fin.sum_univ_four, floodU, floodPrior]

theorem flood_evpi : evpi floodU floodPrior = 207/20 := by
  norm_num [evpi, perfectValue, actNowValue, expectedUtility, maxOver_fin_three,
    Fin.sum_univ_four, floodU, floodPrior]

/-- EVPI is the expected regret of the Bayes action, on this instance. -/
theorem flood_evpi_eq_expectedRegret :
    evpi floodU floodPrior = expectedRegret floodU floodPrior 0 := by
  rw [flood_evpi, flood_expectedRegret.1]

theorem flood_evsi_gauge : evsi floodU floodPrior gauge = 97/20 := by
  norm_num [evsi, valueWithInformation, signalValue, actNowValue, expectedUtility,
    maxOver_fin_three, Fin.sum_univ_four, Fin.sum_univ_two, floodU, floodPrior, gauge]

theorem flood_evsi_garbled : evsi floodU floodPrior (garble gauge flipKernel) = 49/25 := by
  norm_num [evsi, valueWithInformation, signalValue, actNowValue, expectedUtility,
    maxOver_fin_three, Fin.sum_univ_four, Fin.sum_univ_two, floodU, floodPrior, gauge, garble,
    flipKernel]

/-- The garbled gauge is a garbling of the gauge, so T6 applies to it. -/
theorem flood_garbled_isGarblingOf : IsGarblingOf (garble gauge flipKernel) gauge :=
  ⟨flipKernel, flipKernel_stochastic, rfl⟩

/-- Expected utility of each action under each ensemble member (rows members). -/
theorem flood_ensemble_expectedUtility :
    (expectedUtility floodU (floodEnsemble 0) 0 = -39/2 ∧
      expectedUtility floodU (floodEnsemble 0) 1 = -97/4 ∧
      expectedUtility floodU (floodEnsemble 0) 2 = -48) ∧
    (expectedUtility floodU (floodEnsemble 1) 0 = -33 ∧
      expectedUtility floodU (floodEnsemble 1) 1 = -113/4 ∧
      expectedUtility floodU (floodEnsemble 1) 2 = -48) ∧
    (expectedUtility floodU (floodEnsemble 2) 0 = -71/5 ∧
      expectedUtility floodU (floodEnsemble 2) 1 = -227/10 ∧
      expectedUtility floodU (floodEnsemble 2) 2 = -48) := by
  refine ⟨⟨?_, ?_, ?_⟩, ⟨?_, ?_, ?_⟩, ⟨?_, ?_, ?_⟩⟩ <;>
    norm_num [expectedUtility, Fin.sum_univ_four, floodU, floodEnsemble]

theorem flood_lowerExpectedUtility :
    lowerExpectedUtility floodU floodEnsemble 0 = -33 ∧
    lowerExpectedUtility floodU floodEnsemble 1 = -113/4 ∧
    lowerExpectedUtility floodU floodEnsemble 2 = -48 := by
  obtain ⟨⟨a0, a1, a2⟩, ⟨b0, b1, b2⟩, ⟨c0, c1, c2⟩⟩ := flood_ensemble_expectedUtility
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [lowerExpectedUtility, minOver_fin_three, a0, a1, a2, b0, b1, b2, c0, c1, c2]

theorem flood_gammaMaximinValue : gammaMaximinValue floodU floodEnsemble = -113/4 := by
  obtain ⟨h0, h1, h2⟩ := flood_lowerExpectedUtility
  norm_num [gammaMaximinValue, maxOver_fin_three, h0, h1, h2]

/-- "Raise levee" is the unique Γ-maximin action. -/
theorem flood_isGammaMaximin :
    ¬ IsGammaMaximin floodU floodEnsemble 0 ∧ IsGammaMaximin floodU floodEnsemble 1 ∧
      ¬ IsGammaMaximin floodU floodEnsemble 2 := by
  obtain ⟨h0, h1, h2⟩ := flood_lowerExpectedUtility
  simp only [isGammaMaximin_iff, flood_gammaMaximinValue, h0, h1, h2]
  norm_num

/-- The E-admissible set is {do nothing, raise levee}: each is the Bayes action of an ensemble
member, and "relocate" is strictly dominated by "do nothing" under every member, hence not
maximal, hence not E-admissible for any prior in the convex hull. -/
theorem flood_isEAdmissible :
    IsEAdmissible floodU floodEnsemble 0 ∧ IsEAdmissible floodU floodEnsemble 1 ∧
      ¬ IsEAdmissible floodU floodEnsemble 2 := by
  obtain ⟨⟨a0, a1, a2⟩, ⟨b0, b1, b2⟩, ⟨c0, c1, c2⟩⟩ := flood_ensemble_expectedUtility
  refine ⟨?_, ?_, ?_⟩
  · refine IsBayes.isEAdmissible (i := 0) ?_
    unfold IsBayes
    norm_num [forall_fin_three, a0, a1, a2]
  · refine IsBayes.isEAdmissible (i := 1) ?_
    unfold IsBayes
    norm_num [forall_fin_three, b0, b1, b2]
  · intro h
    refine h.isMaximal ⟨0, ?_⟩
    norm_num [forall_fin_three, a0, a2, b0, b2, c0, c2]

/-! ### Two-state epidemic -/

/-- Rows: wait, intervene; columns: low, high. -/
def epidemicU : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 0 | 0, 1 => -100
  | 1, 0 => -30 | 1, 1 => -30

def epidemicPrior : Fin 2 → ℚ
  | 0 => 3/5 | 1 => 2/5

/-- The 80/80 test: signal 0 negative, 1 positive. -/
def epidemicTest : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 4/5 | 0, 1 => 1/5
  | 1, 0 => 1/5 | 1, 1 => 4/5

theorem epidemic_evpi : evpi epidemicU epidemicPrior = 18 := by
  norm_num [evpi, perfectValue, actNowValue, expectedUtility, maxOver_fin_two,
    Fin.sum_univ_two, epidemicU, epidemicPrior]

theorem epidemic_evsi : evsi epidemicU epidemicPrior epidemicTest = 44/5 := by
  norm_num [evsi, valueWithInformation, signalValue, actNowValue, expectedUtility,
    maxOver_fin_two, Fin.sum_univ_two, epidemicU, epidemicPrior, epidemicTest]

/-- Net value of the test at cost 3. -/
theorem epidemic_net : evsi epidemicU epidemicPrior epidemicTest - 3 = 29/5 := by
  rw [epidemic_evsi]
  norm_num

/-! ### Equipment hazard -/

/-- Rows: run, replace; columns: no hazard, hazard. Utilities are minus cost. -/
def equipmentU : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 0 | 0, 1 => -1000
  | 1, 0 => -200 | 1, 1 => -200

def equipmentPrior : Fin 2 → ℚ
  | 0 => 7/10 | 1 => 3/10

/-- The 80/90 test: signal 0 negative, 1 positive. -/
def equipmentTest : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 9/10 | 0, 1 => 1/10
  | 1, 0 => 1/5 | 1, 1 => 4/5

theorem equipment_evpi : evpi equipmentU equipmentPrior = 140 := by
  norm_num [evpi, perfectValue, actNowValue, expectedUtility, maxOver_fin_two,
    Fin.sum_univ_two, equipmentU, equipmentPrior]

theorem equipment_evsi : evsi equipmentU equipmentPrior equipmentTest = 78 := by
  norm_num [evsi, valueWithInformation, signalValue, actNowValue, expectedUtility,
    maxOver_fin_two, Fin.sum_univ_two, equipmentU, equipmentPrior, equipmentTest]

/-- Net value of the test at cost 30. -/
theorem equipment_net : evsi equipmentU equipmentPrior equipmentTest - 30 = 48 := by
  rw [equipment_evsi]
  norm_num

theorem equipment_expectedRegret :
    expectedRegret equipmentU equipmentPrior 0 = 240 ∧
    expectedRegret equipmentU equipmentPrior 1 = 140 := by
  constructor <;>
    norm_num [expectedRegret, regret, maxOver_fin_two, Fin.sum_univ_two, equipmentU,
      equipmentPrior]

theorem equipment_maxRegret :
    maxRegret equipmentU 0 = 800 ∧ maxRegret equipmentU 1 = 200 := by
  constructor <;> norm_num [maxRegret, regret, maxOver_fin_two, equipmentU]

end DecisionAnalysisProofs.Examples
