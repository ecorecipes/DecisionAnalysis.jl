import DecisionAnalysisProofs.Finite.Extrema

/-!
# DecisionAnalysisProofs.Finite.Problem

The finite normal-form decision problem of `DecisionAnalysis.jl` (`src/payoff.jl`,
`src/problem.jl`, `src/regret.jl`, `src/voi.jl`): a payoff table, a prior, and the quantities
computed from them.

## Conventions

* Actions `A`, states `Θ` and signals `Z` are finite types; `A` is nonempty. A payoff table is
  `u : A → Θ → K` (`PayoffTable`, rows are actions, columns are states), a prior is
  `p : Θ → K` (`FiniteBelief`), an experiment is `e : Θ → Z → K` with `e θ z = P(z | θ)`
  (`Experiment`; the Julia likelihood matrix has rows signals and columns states, so
  `e θ z = likelihood[z, θ]`).
* `K` is any linearly ordered field. The theorems are used at `K = ℝ`; the worked examples and
  the certificate checker instantiate the *same* definitions at `K = ℚ`.
* Nonnegativity and normalisation are separate predicates (`NonnegPrior`, `NormalisedPrior`,
  `NonnegKernel`, `NormalisedKernel`, `StochasticKernel`) and are hypotheses of exactly the
  theorems that need them. At `K = ℝ` the kernel predicates are those of
  `FiniteKernelsProofs.Finite.Kernel` (`Theory/FiniteKernels.lean`).
* The value with information is in **joint form**, `Σ_z max_a Σ_θ p θ · e θ z · u a θ`: no
  posterior, no division, no case split on signals of probability zero.

## Definitions

| Lean | Julia |
|:--|:--|
| `expectedUtility`, `actNowValue`, `IsBayes` | `decide(problem, MaxExpectedUtility())` |
| `valueWithInformation`, `evsi` | `value_with_information`, `evsi` |
| `perfectValue`, `evpi` | `evpi` |
| `partialValue`, `evppi` | `evppi` (exact method, focal quantity `f θ`) |
| `regret`, `expectedRegret`, `maxRegret` | `regret_table`, `expected_regret`, `max_regret` |
| `IsMinimaxRegret`, `minimaxRegretValue` | `decide(problem, MinimaxRegret())` |
| `worstCase` | `decide(problem, Maximin())` |

Basic facts: **`exists_isBayes`** (a Bayes action exists), **`isBayes_iff`**,
**`sum_signalValue`** (the joint-form objectives sum to the expected utility),
**`expectedRegret_eq`**, **`partialValue_eq_valueWithInformation`**.

Not claimed: anything about `Float64` arithmetic, tolerances, or tie-breaking in Julia.
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z C : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z] [Fintype C]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### Predicates -/

/-- Every prior weight is nonnegative. -/
def NonnegPrior (p : Θ → K) : Prop := ∀ θ, 0 ≤ p θ

/-- The prior weights sum to one. -/
def NormalisedPrior (p : Θ → K) : Prop := ∑ θ, p θ = 1

/-- Every entry of a kernel is nonnegative. -/
def NonnegKernel (e : Θ → Z → K) : Prop := ∀ θ z, 0 ≤ e θ z

/-- Every row `e θ` of a kernel sums to one. -/
def NormalisedKernel (e : Θ → Z → K) : Prop := ∀ θ, ∑ z, e θ z = 1

/-- A stochastic kernel: nonnegative and normalised. -/
def StochasticKernel (e : Θ → Z → K) : Prop := NonnegKernel e ∧ NormalisedKernel e

/-! ### Acting now -/

/-- Expected utility of action `a`: `Σ_θ p θ · u a θ` (one entry of `payoff * prior`). -/
def expectedUtility (u : A → Θ → K) (p : Θ → K) (a : A) : K := ∑ θ, p θ * u a θ

/-- The value of acting now: the maximum expected utility. -/
def actNowValue (u : A → Θ → K) (p : Θ → K) : K := maxOver (expectedUtility u p)

/-- `a` is a Bayes action: it maximises expected utility (`MaxExpectedUtility`). -/
def IsBayes (u : A → Θ → K) (p : Θ → K) (a : A) : Prop :=
  ∀ a', expectedUtility u p a' ≤ expectedUtility u p a

theorem expectedUtility_le_actNowValue (u : A → Θ → K) (p : Θ → K) (a : A) :
    expectedUtility u p a ≤ actNowValue u p := le_maxOver _ a

theorem isBayes_iff (u : A → Θ → K) (p : Θ → K) (a : A) :
    IsBayes u p a ↔ expectedUtility u p a = actNowValue u p := by
  constructor
  · intro h
    exact (maxOver_eq_of a rfl h).symm
  · intro h a'
    rw [h]
    exact expectedUtility_le_actNowValue u p a'

/-- The act-now value is attained: a Bayes action exists. -/
theorem exists_isBayes (u : A → Θ → K) (p : Θ → K) : ∃ a, IsBayes u p a := by
  obtain ⟨a, ha⟩ := exists_eq_maxOver (expectedUtility u p)
  exact ⟨a, (isBayes_iff u p a).mpr ha.symm⟩

/-! ### Acting after a signal -/

/-- The joint-form objective after signal `z`: `Σ_θ p θ · e θ z · u a θ`, the posterior
expected utility of `a` multiplied by the probability of `z`. -/
def signalValue (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) (z : Z) (a : A) : K :=
  ∑ θ, p θ * e θ z * u a θ

/-- The marginal probability of signal `z`. -/
def signalProbability (p : Θ → K) (e : Θ → Z → K) (z : Z) : K := ∑ θ, p θ * e θ z

/-- `value_with_information`: `Σ_z max_a Σ_θ p θ · e θ z · u a θ`. -/
def valueWithInformation (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) : K :=
  ∑ z, maxOver (signalValue u p e z)

/-- `evsi`: the expected value of sample information of the experiment `e`. -/
def evsi (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) : K :=
  valueWithInformation u p e - actNowValue u p

/-- The value of deciding after the state is revealed: `Σ_θ p θ · max_a u a θ`. -/
def perfectValue (u : A → Θ → K) (p : Θ → K) : K := ∑ θ, p θ * maxOver fun a => u a θ

/-- `evpi`: the expected value of perfect information. -/
def evpi (u : A → Θ → K) (p : Θ → K) : K := perfectValue u p - actNowValue u p

/-- The deterministic experiment that reveals `f θ` (`deterministic` in FiniteKernels.jl). -/
def deterministic [DecidableEq C] (f : Θ → C) : Θ → C → K := fun θ c => if c = f θ then 1 else 0

/-- The value of deciding after `f θ` is revealed:
`Σ_c max_a Σ_{θ : f θ = c} p θ · u a θ`. -/
def partialValue [DecidableEq C] (u : A → Θ → K) (p : Θ → K) (f : Θ → C) : K :=
  ∑ c, maxOver fun a => ∑ θ, if f θ = c then p θ * u a θ else 0

/-- `evppi`: the expected value of partial perfect information about `f θ`. -/
def evppi [DecidableEq C] (u : A → Θ → K) (p : Θ → K) (f : Θ → C) : K :=
  partialValue u p f - actNowValue u p

/-- When the experiment is normalised, the joint-form objectives sum over signals to the
expected utility. -/
theorem sum_signalValue (u : A → Θ → K) (p : Θ → K) {e : Θ → Z → K} (he : NormalisedKernel e)
    (a : A) : ∑ z, signalValue u p e z a = expectedUtility u p a := by
  unfold signalValue expectedUtility
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun θ _ => ?_
  rw [← Finset.sum_mul, ← Finset.mul_sum, he θ, mul_one]

/-- Partial perfect information about `f θ` is the experiment `deterministic f`. -/
theorem partialValue_eq_valueWithInformation [DecidableEq C] (u : A → Θ → K) (p : Θ → K)
    (f : Θ → C) : partialValue u p f = valueWithInformation u p (deterministic f) := by
  unfold partialValue valueWithInformation
  refine Finset.sum_congr rfl fun c _ => congrArg maxOver (funext fun a => ?_)
  unfold signalValue
  refine Finset.sum_congr rfl fun θ _ => ?_
  by_cases h : f θ = c
  · subst h
    simp [deterministic]
  · have h' : ¬ c = f θ := fun h' => h h'.symm
    simp [deterministic, h, h']

theorem evppi_eq_evsi [DecidableEq C] (u : A → Θ → K) (p : Θ → K) (f : Θ → C) :
    evppi u p f = evsi u p (deterministic f) := by
  unfold evppi evsi
  rw [partialValue_eq_valueWithInformation]

/-! ### Regret -/

/-- `regret_table`: `R a θ = max_a' u a' θ - u a θ`. -/
def regret (u : A → Θ → K) (a : A) (θ : Θ) : K := (maxOver fun a' => u a' θ) - u a θ

/-- `expected_regret`: `Σ_θ p θ · R a θ`. -/
def expectedRegret (u : A → Θ → K) (p : Θ → K) (a : A) : K := ∑ θ, p θ * regret u a θ

/-- `max_regret`: the largest regret of `a` over states. -/
def maxRegret [Nonempty Θ] (u : A → Θ → K) (a : A) : K := maxOver (regret u a)

/-- The minimax regret value. -/
def minimaxRegretValue [Nonempty Θ] (u : A → Θ → K) : K := minOver (maxRegret u)

/-- `a` is a minimax-regret action (`MinimaxRegret`). -/
def IsMinimaxRegret [Nonempty Θ] (u : A → Θ → K) (a : A) : Prop :=
  ∀ a', maxRegret u a ≤ maxRegret u a'

/-- The worst-case payoff of `a` over states (the score used by `Maximin`). -/
def worstCase [Nonempty Θ] (u : A → Θ → K) (a : A) : K := minOver (u a)

theorem regret_nonneg (u : A → Θ → K) (a : A) (θ : Θ) : 0 ≤ regret u a θ :=
  sub_nonneg.mpr (le_maxOver (fun a' => u a' θ) a)

/-- Expected regret is the perfect-information value minus the expected utility. -/
theorem expectedRegret_eq (u : A → Θ → K) (p : Θ → K) (a : A) :
    expectedRegret u p a = perfectValue u p - expectedUtility u p a := by
  simp [expectedRegret, regret, perfectValue, expectedUtility, mul_sub, Finset.sum_sub_distrib]

theorem isMinimaxRegret_iff [Nonempty Θ] (u : A → Θ → K) (a : A) :
    IsMinimaxRegret u a ↔ maxRegret u a = minimaxRegretValue u := by
  constructor
  · intro h
    exact (minOver_eq_of a rfl h).symm
  · intro h a'
    rw [h]
    exact minOver_le _ a'

end DecisionAnalysisProofs
