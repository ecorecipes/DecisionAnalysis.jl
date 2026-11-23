import DecisionAnalysisProofs.Finite.Problem
import Mathlib.Tactic.NormNum

/-!
# DecisionAnalysisProofs.Finite.MinimaxRegretIIA

**T8**: minimax regret (`decide(problem, MinimaxRegret())`, `src/rules.jl`) violates
independence of irrelevant alternatives. An explicit counterexample over `ℚ`, with two states:

| action | state 0 | state 1 | max regret among {0, 1} | max regret among {0, 1, 2} |
|:--|--:|--:|--:|--:|
| 0 | 10 | 0 | 5 | 12 |
| 1 | 4 | 5 | 6 | 7 |
| 2 | 0 | 12 | not offered | 10 |

With actions 0 and 1 only, action 0 is the unique minimax-regret action. Adding action 2,
which is not itself chosen, makes action 1 the unique minimax-regret action.

* **`isMinimaxRegret_iiaSmall_iff`**, **`isMinimaxRegret_iiaLarge_iff`**: the two choice sets.
* **`minimaxRegret_violates_iia`**: the statement of the violation.

Everything is evaluated by `simp` and `norm_num` on the definitions of `Finite/Problem.lean`.
-/

set_option autoImplicit false

namespace DecisionAnalysisProofs

/-- The three-action payoff table of the counterexample. -/
def iiaLarge : Fin 3 → Fin 2 → ℚ
  | 0, 0 => 10 | 0, 1 => 0
  | 1, 0 => 4 | 1, 1 => 5
  | 2, 0 => 0 | 2, 1 => 12

/-- The same table with the third action removed. -/
def iiaSmall : Fin 2 → Fin 2 → ℚ
  | 0, 0 => 10 | 0, 1 => 0
  | 1, 0 => 4 | 1, 1 => 5

/-- The small table is the large one restricted to the first two actions. -/
theorem iiaSmall_eq (a : Fin 2) (θ : Fin 2) : iiaSmall a θ = iiaLarge a.castSucc θ := by
  fin_cases a <;> fin_cases θ <;> rfl

theorem maxRegret_iiaSmall :
    maxRegret iiaSmall 0 = 5 ∧ maxRegret iiaSmall 1 = 6 := by
  constructor <;>
    norm_num [maxRegret, regret, maxOver_fin_two, iiaSmall]

theorem maxRegret_iiaLarge :
    maxRegret iiaLarge 0 = 12 ∧ maxRegret iiaLarge 1 = 7 ∧ maxRegret iiaLarge 2 = 10 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [maxRegret, regret, maxOver_fin_two, maxOver_fin_three, iiaLarge]

/-- Among the first two actions, action 0 is the unique minimax-regret action. -/
theorem isMinimaxRegret_iiaSmall_iff (a : Fin 2) : IsMinimaxRegret iiaSmall a ↔ a = 0 := by
  obtain ⟨h0, h1⟩ := maxRegret_iiaSmall
  revert a
  unfold IsMinimaxRegret
  norm_num [Fin.forall_fin_two, h0, h1]

/-- Among all three actions, action 1 is the unique minimax-regret action. -/
theorem isMinimaxRegret_iiaLarge_iff (a : Fin 3) : IsMinimaxRegret iiaLarge a ↔ a = 1 := by
  obtain ⟨h0, h1, h2⟩ := maxRegret_iiaLarge
  revert a
  unfold IsMinimaxRegret
  norm_num [forall_fin_three, h0, h1, h2]
  decide

/-- **T8.** Minimax regret violates independence of irrelevant alternatives: there are a payoff
table with three actions and its restriction to the first two such that action 0 is chosen and
action 1 rejected from the pair, while from the triple action 1 is chosen and both action 0
and the added action 2 are rejected. -/
theorem minimaxRegret_violates_iia :
    ∃ (u : Fin 3 → Fin 2 → ℚ) (v : Fin 2 → Fin 2 → ℚ),
      (∀ a θ, v a θ = u a.castSucc θ) ∧
      IsMinimaxRegret v 0 ∧ ¬ IsMinimaxRegret v 1 ∧
      IsMinimaxRegret u 1 ∧ ¬ IsMinimaxRegret u 0 ∧ ¬ IsMinimaxRegret u 2 := by
  refine ⟨iiaLarge, iiaSmall, iiaSmall_eq, ?_, ?_, ?_, ?_, ?_⟩
  · exact (isMinimaxRegret_iiaSmall_iff 0).mpr rfl
  · rw [isMinimaxRegret_iiaSmall_iff]; decide
  · exact (isMinimaxRegret_iiaLarge_iff 1).mpr rfl
  · rw [isMinimaxRegret_iiaLarge_iff]; decide
  · rw [isMinimaxRegret_iiaLarge_iff]; decide

end DecisionAnalysisProofs
