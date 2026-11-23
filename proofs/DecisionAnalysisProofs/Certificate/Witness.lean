import DecisionAnalysisProofs.Certificate.Check
import DecisionAnalysisProofs.Finite.Credal

/-!
# DecisionAnalysisProofs.Certificate.Witness

Checkers for the witnesses that `DecisionAnalysis.jl` obtains from linear programmes
(`src/lp.jl`), so that the answers can be trusted without trusting the solver.

## Garbling witness (`is_garbling_of`, `src/experiments.jl`)

`is_garbling_of(e′, e)` returns a matrix `G` with `G[z′, z] = P(z′ | z)` and
`G * L_e = L_e′`. Given the two likelihoods and `G`, `RawGarbling.Accepts` checks that the
three matrices are rectangular and of matching sizes, that `e` and `G` are column-stochastic,
and that the product is `L_e′` entry for entry.

* **`GarblingWitnessOk.isGarblingOf`**, **`RawGarbling.accepts_sound`**: an accepted witness
  proves `IsGarblingOf e′ e`, and therefore (T6) that `e′` is worth no more than `e` for every
  payoff table and every prior over `ℚ`.

## E-admissibility witness (`decide(problem, EAdmissible())`, `src/rules.jl`)

Given the payoff table, the extreme points of the credal set, weights on them, the mixture
prior and an action, `RawEAdmissible.Accepts` checks that the extreme points are probability
vectors, the weights are nonnegative and sum to one, the prior is the stated mixture, and the
action is Bayes under it.

* **`EAdmissibleWitnessOk.isEAdmissible`**, **`RawEAdmissible.accepts_sound`**: an accepted
  witness proves `IsEAdmissible`, hence `IsMaximal`.

Both checks are sound and not complete: a rejected witness says nothing about whether another
witness exists. Completeness needs the Blackwell converse (S1) and the mixed-dominance
characterisation (T12b); see `Roadmap.lean`.

Not claimed: correctness of the JSON parser in `CheckCertificate.lean`, which is trusted.
-/

set_option autoImplicit false

namespace DecisionAnalysisProofs.Certificate

open DecisionAnalysisProofs

/-! ### Garbling -/

/-- `g` is stochastic and `e'` is `e` followed by `g`. -/
def GarblingWitnessOk {s z z' : ℕ} (e : Fin s → Fin z → ℚ) (e' : Fin s → Fin z' → ℚ)
    (g : Fin z → Fin z' → ℚ) : Prop :=
  StochasticKernel g ∧ e' = garble e g

instance {s z z' : ℕ} (e : Fin s → Fin z → ℚ) (e' : Fin s → Fin z' → ℚ)
    (g : Fin z → Fin z' → ℚ) : Decidable (GarblingWitnessOk e e' g) := by
  unfold GarblingWitnessOk; infer_instance

/-- **Soundness of the garbling witness.** -/
theorem GarblingWitnessOk.isGarblingOf {s z z' : ℕ} {e : Fin s → Fin z → ℚ}
    {e' : Fin s → Fin z' → ℚ} {g : Fin z → Fin z' → ℚ} (h : GarblingWitnessOk e e' g) :
    IsGarblingOf e' e :=
  ⟨g, h.1, h.2⟩

/-- A decoded `"garbling-witness/1"` document. -/
structure RawGarbling where
  /-- The number of states (columns of both likelihoods). -/
  nStates : ℕ
  /-- The likelihood of `e`: rows are signals, columns are states. -/
  experiment : List (List ℚ)
  /-- The likelihood of `e′`. -/
  garbled : List (List ℚ)
  /-- The witness `G[z′][z] = P(z′ | z)`: rows are signals of `e′`, columns signals of `e`. -/
  garbling : List (List ℚ)

/-- The experiment `e`. -/
def RawGarbling.e (r : RawGarbling) : Fin r.nStates → Fin r.experiment.length → ℚ :=
  kernelOf r.experiment r.nStates r.experiment.length

/-- The experiment `e′`. -/
def RawGarbling.e' (r : RawGarbling) : Fin r.nStates → Fin r.garbled.length → ℚ :=
  kernelOf r.garbled r.nStates r.garbled.length

/-- The kernel `g z z′ = G[z′][z]`. -/
def RawGarbling.g (r : RawGarbling) : Fin r.experiment.length → Fin r.garbled.length → ℚ :=
  kernelOf r.garbling r.experiment.length r.garbled.length

/-- What the garbling-witness checker decides. -/
def RawGarbling.Accepts (r : RawGarbling) : Prop :=
  rect r.experiment r.experiment.length r.nStates = true ∧
  rect r.garbled r.garbled.length r.nStates = true ∧
  rect r.garbling r.garbled.length r.experiment.length = true ∧
  StochasticKernel r.e ∧ GarblingWitnessOk r.e r.e' r.g

instance (r : RawGarbling) : Decidable r.Accepts := by
  unfold RawGarbling.Accepts; infer_instance

/-- The garbling-witness checker run by `lake exe check_certificate`. -/
def checkGarbling (r : RawGarbling) : Bool := decide r.Accepts

/-- **Soundness of the garbling-witness checker.** If it accepts, `e′` is a garbling of `e`,
`e′` is itself stochastic, and `e′` is worth no more than `e` in every decision problem. -/
theorem RawGarbling.accepts_sound {r : RawGarbling} (h : checkGarbling r = true) :
    IsGarblingOf r.e' r.e ∧ StochasticKernel r.e' ∧
      ∀ (n : ℕ) (u : Fin (n + 1) → Fin r.nStates → ℚ) (p : Fin r.nStates → ℚ),
        valueWithInformation u p r.e' ≤ valueWithInformation u p r.e := by
  have h' : r.Accepts := of_decide_eq_true h
  obtain ⟨-, -, -, he, hw⟩ := h'
  have hg := hw.isGarblingOf
  refine ⟨hg, ?_, fun n u p => hg.valueWithInformation_le u p⟩
  rw [hw.2]
  exact he.garble hw.1

/-! ### E-admissibility -/

/-- `q` is the mixture of the family `P` with probability weights `w`, and `a` is Bayes under
`q`. -/
def EAdmissibleWitnessOk {n k s : ℕ} (u : Fin n → Fin s → ℚ) (P : Fin k → Fin s → ℚ)
    (w : Fin k → ℚ) (q : Fin s → ℚ) (a : Fin n) : Prop :=
  (∀ i, 0 ≤ w i) ∧ ∑ i, w i = 1 ∧ q = mixture w P ∧ IsBayes u q a

instance {n k s : ℕ} (u : Fin n → Fin s → ℚ) (P : Fin k → Fin s → ℚ) (w : Fin k → ℚ)
    (q : Fin s → ℚ) (a : Fin n) : Decidable (EAdmissibleWitnessOk u P w q a) := by
  unfold EAdmissibleWitnessOk; infer_instance

/-- **Soundness of the E-admissibility witness.** -/
theorem EAdmissibleWitnessOk.isEAdmissible {n k s : ℕ} {u : Fin n → Fin s → ℚ}
    {P : Fin k → Fin s → ℚ} {w : Fin k → ℚ} {q : Fin s → ℚ} {a : Fin n}
    (h : EAdmissibleWitnessOk u P w q a) : IsEAdmissible u P a :=
  ⟨w, h.1, h.2.1, h.2.2.1 ▸ h.2.2.2⟩

/-- A decoded `"e-admissible-witness/1"` document. -/
structure RawEAdmissible where
  /-- The payoff table: rows are actions, columns are states. -/
  payoff : List (List ℚ)
  /-- The extreme points of the credal set: one prior per row. -/
  priors : List (List ℚ)
  /-- The convex-hull weights, one per extreme point. -/
  weights : List ℚ
  /-- The witness prior, claimed to be the mixture. -/
  prior : List ℚ
  /-- The zero-based index of the action. -/
  action : ℕ

/-- The payoff table. -/
def RawEAdmissible.u (r : RawEAdmissible) : Fin r.payoff.length → Fin r.prior.length → ℚ :=
  table r.payoff r.payoff.length r.prior.length

/-- The extreme points. -/
def RawEAdmissible.P (r : RawEAdmissible) : Fin r.weights.length → Fin r.prior.length → ℚ :=
  table r.priors r.weights.length r.prior.length

/-- What the E-admissibility-witness checker decides. -/
def RawEAdmissible.Accepts (r : RawEAdmissible) : Prop :=
  rect r.payoff r.payoff.length r.prior.length = true ∧
  rect r.priors r.weights.length r.prior.length = true ∧
  (∀ i, NonnegPrior (r.P i) ∧ NormalisedPrior (r.P i)) ∧
  ∃ h : r.action < r.payoff.length,
    EAdmissibleWitnessOk r.u r.P (vec r.weights r.weights.length)
      (vec r.prior r.prior.length) ⟨r.action, h⟩

instance (r : RawEAdmissible) : Decidable r.Accepts := by
  unfold RawEAdmissible.Accepts; infer_instance

/-- The E-admissibility-witness checker run by `lake exe check_certificate`. -/
def checkEAdmissible (r : RawEAdmissible) : Bool := decide r.Accepts

/-- **Soundness of the E-admissibility-witness checker.** If it accepts, the action is
E-admissible for the credal set generated by the stated extreme points, and hence maximal. -/
theorem RawEAdmissible.accepts_sound {r : RawEAdmissible} (h : checkEAdmissible r = true) :
    ∃ ha : r.action < r.payoff.length,
      IsEAdmissible r.u r.P ⟨r.action, ha⟩ ∧ IsMaximal r.u r.P ⟨r.action, ha⟩ := by
  have h' : r.Accepts := of_decide_eq_true h
  obtain ⟨-, -, -, ha, hw⟩ := h'
  exact ⟨ha, hw.isEAdmissible, hw.isEAdmissible.isMaximal⟩

end DecisionAnalysisProofs.Certificate
