import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

/-!
# DecisionAnalysisProofs.Finite.SurveyOrAct

The arithmetic behind `survey_or_act` (`src/survey.jl`): a management unit is acted on now or
surveyed first, and the two options are compared by value per unit of expected cost (Raymond
et al. 2020, J Appl Ecol 57:819-830). Everything is over an ordered field, so the statements
hold for the exact rational arithmetic of the package.

## What each result backs or shapes

* **`survey_first_iff`**: one action worth 1 where the species is present (prior `p`) at
  cost `C`, and a survey of cost `M` that finds a present species with probability `d` and
  never an absent one. Acting now is worth `p / C` per unit of cost, surveying and acting
  only where the species is found `d p / (d p C + M)`, and the second is strictly larger
  exactly when `M < d C (1 − p)`. Backs the closed form in the docstring and the Julia test
  that compares the choice of `survey_or_act` with this inequality on a grid.
* **`survey_first_tie`**: on the boundary `M = d C (1 − p)` the two ratios are equal. Shapes
  the package: the choice needs a stated tie rule (act now).
* **`survey_same_action_lt`**: surveying and then taking the same action whatever is seen has
  the value of acting now at a higher cost, so it is strictly worse per unit of cost. Backs
  the sentence in the docstring that a survey is chosen only when some outcome leads to a
  cheaper action, and is the reason the policy search may start from the baseline.
* **`ratio_le_of_parametric`**: the stopping test of the policy search (Dinkelbach). If a
  policy `π₀` with ratio `λ = g π₀ / c π₀` maximises `g π − λ c π`, it maximises `g π / c π`.
  In the package `g` and `c` are sums over signals, so the parametric maximum is taken
  signal by signal; the Julia test checks the result against every policy by enumeration.

Not claimed: that ranking units by value per cost fills a budget optimally
(`select_under_budget` is a heuristic for a knapsack problem, and the docstring says so), or
anything about realised, as opposed to expected, cost.
-/

set_option autoImplicit false

namespace DecisionAnalysisProofs

variable {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- Survey first, and act only where the species is found, beats acting now per unit of cost
exactly when the survey costs less than `d C (1 − p)`. -/
theorem survey_first_iff {p C M d : K} (hp : 0 < p) (hC : 0 < C) (hM : 0 < M) (hd : 0 < d) :
    p / C < d * p / (d * p * C + M) ↔ M < d * C * (1 - p) := by
  have hden : 0 < d * p * C + M := by positivity
  rw [div_lt_div_iff₀ hC hden]
  constructor
  · intro h
    have h' : p * (d * p * C + M) < p * (d * C) := by nlinarith
    have := lt_of_mul_lt_mul_left h' hp.le
    nlinarith
  · intro h
    have h' : p * (d * p * C + M) < p * (d * C) :=
      mul_lt_mul_of_pos_left (by nlinarith) hp
    nlinarith

/-- On the boundary the two options are worth the same per unit of cost. -/
theorem survey_first_tie {p C M d : K} (hp : 0 < p) (hC : 0 < C) (hM : 0 < M) (hd : 0 < d)
    (h : M = d * C * (1 - p)) : p / C = d * p / (d * p * C + M) := by
  have hden : 0 < d * p * C + M := by positivity
  rw [div_eq_div_iff hC.ne' hden.ne', h]
  ring

/-- Surveying and then taking the same action whatever is seen is strictly worse, per unit of
cost, than taking that action now. -/
theorem survey_same_action_lt {v C M : K} (hv : 0 < v) (hC : 0 < C) (hM : 0 < M) :
    v / (C + M) < v / C :=
  div_lt_div_of_pos_left hv hC (by linarith)

/-- Dinkelbach's stopping test: a policy that maximises `g − λ c` at its own ratio `λ`
maximises the ratio `g / c`. -/
theorem ratio_le_of_parametric {P : Type} (g c : P → K) (π₀ : P) (hc : ∀ π, 0 < c π)
    (h : ∀ π, g π - g π₀ / c π₀ * c π ≤ g π₀ - g π₀ / c π₀ * c π₀) (π : P) :
    g π / c π ≤ g π₀ / c π₀ := by
  rw [div_le_iff₀ (hc π)]
  have h0 : g π₀ / c π₀ * c π₀ = g π₀ := div_mul_cancel₀ _ (hc π₀).ne'
  have := h π
  linarith

end DecisionAnalysisProofs
